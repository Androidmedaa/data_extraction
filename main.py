"""
Strict document-structure extraction engine.
Transforms Unstructured PDF parser elements into structural JSON only.
No semantic interpretation, no normalization, no text changes.

NOT: Bu çıktı ham ingestion çıktısıdır; embedding öncesi filtreleme gerektirir.
     RAG için hazır birimler: get_embedding_chunks(result) kullanın.

KULLANIM (3 yol):

1) Komut satırından PDF dosyası:
   python main.py makale.pdf
   python main.py makale.pdf -o cikti.json

2) Python içinde PDF'ten:
   from main import pdf_to_structure, get_embedding_chunks
   result = pdf_to_structure("makale.pdf")
   chunks = get_embedding_chunks(result)   # RAG embedding adayları

3) Zaten elinde element listesi varsa:
   from main import extract_structure, to_json, get_embedding_chunks
   result = extract_structure(elements)
   json_str = to_json(elements)
   chunks = get_embedding_chunks(result)
"""

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

# Unstructured category → output block type (schema: title | paragraph | list_item | table | header | footer | other)
CATEGORY_TO_TYPE = {
    "Title": "title",
    "NarrativeText": "paragraph",
    "ListItem": "list_item",
    "Table": "table",
    "Header": "header",
    "Footer": "footer",
}


def _block_type(category: str) -> str:
    """Map category to schema type. Unknown → other."""
    if category in CATEGORY_TO_TYPE:
        return CATEGORY_TO_TYPE[category]
    return "other"


def _slug(text: str) -> str:
    """Başlıktan section_id üretir (Türkçe uyumlu, küçük harf, alt çizgi)."""
    if not text or not isinstance(text, str):
        return "basliksiz"
    tr_map = str.maketrans("ıİğĞüÜşŞöÖçÇ ", "iIgGuUsSoOcC_")
    s = text.strip().translate(tr_map).lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "basliksiz"


def _normalize_text(text: str) -> str:
    """Arama/embedding için: boşluk birleştir, trim, NFC."""
    if not text or not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFC", text)
    s = " ".join(s.split())
    return s.strip()


def _hierarchy_level(block_type: str) -> int:
    """title=1, list_item=2, paragraph/table/other=3, header/footer=4."""
    if block_type == "title":
        return 1
    if block_type == "list_item":
        return 2
    if block_type in ("header", "footer"):
        return 4
    return 3


# Embedding'e GİRMEMESİ gerekenler: footer, sayfa no, tarih, copyright, link, içindekiler, UI ifadeleri, boilerplate
_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4})\b",
    re.IGNORECASE,
)
_COPYRIGHT_PATTERN = re.compile(
    r"©|copyright|tüm hakları|all rights reserved|©\s*\d{4}|her hakkı",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://|www\.|\.com\b|\.org\b|\.net\b", re.IGNORECASE)
_TOC_LINE_PATTERN = re.compile(r"^\s*\d+[.)]\s*.{1,100}$")  # "1. Giriş" / "2) Yöntem"

_ICINDEKILER_MARKERS = ("içindekiler", "contents", "table of contents", "index")
_UI_INSTRUCTION_MARKERS = (
    "tıklayın", "tıklanır", "tıklanacak", "butona", "butonuna", "menüden", "seçin", "girin",
    "ekran açılır", "açılır", "penceresi açılır", "görüntülenir", "görünür", "sayfa açılır",
    "click", "select", "choose", "enter", "opens", "displayed", "shown",
)


def _is_noise(
    block_type: str,
    text: str,
    section_title: str = "",
) -> bool:
    """Embedding'e alınmaması gereken bloklar: footer, sayfa no, tarih, copyright, link, içindekiler, UI ifadeleri."""
    if block_type in ("footer", "header"):
        return True
    t = text.strip()
    if not t:
        return True
    if len(t) <= 2 and t in (".", "..", "-", "–", "—"):
        return True
    if t.isdigit():
        return True

    lower = t.lower()
    # Tarih (tek başına veya baskın)
    if _DATE_PATTERN.search(t) and len(t) < 80:
        return True
    # Copyright
    if _COPYRIGHT_PATTERN.search(t):
        return True
    # Link / URL ağırlıklı
    if _URL_PATTERN.search(t) and sum(1 for c in t if c in " .,;") < 3:
        return True
    # İçindekiler bölümü veya TOC satırı ("1. Giriş", "2. Yöntem")
    if section_title and section_title.strip().lower() in _ICINDEKILER_MARKERS:
        return True
    if "içindekiler" in lower and len(t) < 150:
        return True
    if _TOC_LINE_PATTERN.match(t) and len(t) < 120:
        return True
    # Kısa UI / ekran anlatımı
    if len(t) < 120 and any(m in lower for m in _UI_INSTRUCTION_MARKERS):
        return True

    return False


def _bbox_from_element(el: dict[str, Any]) -> list[float] | None:
    """Extract [x1, y1, x2, y2] if available; else None."""
    bbox = el.get("bbox") or el.get("coordinates") or el.get("bounding_box")
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    if isinstance(bbox, dict):
        x1 = bbox.get("x1") or bbox.get("left")
        y1 = bbox.get("y1") or bbox.get("top")
        x2 = bbox.get("x2") or bbox.get("right")
        y2 = bbox.get("y2") or bbox.get("bottom")
        if None not in (x1, y1, x2, y2):
            return [float(x1), float(y1), float(x2), float(y2)]
    return None


def extract_structure(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Transform a list of Unstructured-style elements into the strict output schema.
    Preserves text, page, bbox; adds RAG-oriented fields: section_id, section_title,
    hierarchy_level, chunk_id, block_index_in_page, is_noise, normalized_text.
    """
    blocks = []
    current_section_id = "basliksiz"
    current_section_title = ""
    page_counter: dict[int, int] = {}
    section_block_index: dict[str, int] = {}

    for i, el in enumerate(elements):
        category = el.get("category") or el.get("type") or ""
        if isinstance(category, str):
            category = category.strip()
        block_type = _block_type(category)

        text = el.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        page = el.get("page_number") or el.get("page")
        if page is None:
            page = 0
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = 0

        bbox = _bbox_from_element(el)

        if block_type == "title":
            current_section_id = _slug(text)
            current_section_title = text.strip()

        section_block_index[current_section_id] = section_block_index.get(current_section_id, 0) + 1
        idx_in_section = section_block_index[current_section_id]
        chunk_id = f"page_{page}_section_{current_section_id}_{idx_in_section:03d}"

        block_index_in_page = page_counter.get(page, 0) + 1
        page_counter[page] = block_index_in_page

        norm_text = _normalize_text(text)
        noise = _is_noise(block_type, text, current_section_title)

        blocks.append({
            "block_id": f"block_{i}",
            "type": block_type,
            "text": text,
            "page": page,
            "bbox": bbox,
            "section_id": current_section_id or "basliksiz",
            "section_title": current_section_title or "",
            "hierarchy_level": _hierarchy_level(block_type),
            "chunk_id": chunk_id,
            "block_index_in_page": block_index_in_page,
            "is_noise": noise,
            "normalized_text": norm_text,
        })

    # Tekrar eden boilerplate: aynı metin 3+ farklı sayfada varsa embedding'e alma
    _mark_boilerplate_noise(blocks)

    return {
        "document_type": "unknown",
        "blocks": blocks,
    }


def _mark_boilerplate_noise(blocks: list[dict[str, Any]]) -> None:
    """Aynı normalized_text 3+ sayfada geçiyorsa is_noise=True yap (her sayfada tekrarlayan açıklama)."""
    text_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, b in enumerate(blocks):
        nt = b.get("normalized_text") or ""
        if len(nt) >= 25:  # Çok kısa tekrarları sayma
            text_to_indices[nt].append(idx)
    for nt, indices in text_to_indices.items():
        pages = {blocks[i]["page"] for i in indices}
        if len(pages) >= 3:
            for i in indices:
                blocks[i]["is_noise"] = True


def to_json(elements: list[dict[str, Any]], *, ensure_ascii: bool = False) -> str:
    """Return the structural JSON as a string (no extra text/comments)."""
    return json.dumps(extract_structure(elements), ensure_ascii=ensure_ascii, indent=2)


# ---- ADIM 1–3: İçindekiler çöpe, başlık+paragraf birlikte, sadece içerik olan bölümler ----

def _should_discard_toc(block: dict[str, Any]) -> bool:
    """ADIM 1: type=other + (..... / sayfa no / İçindekiler) → çöpe at."""
    if block.get("type") != "other":
        return False
    t = (block.get("text") or "").strip()
    if not t:
        return True
    if t.isdigit():
        return True
    lower = t.lower()
    if "içindekiler" in lower or "contents" in lower:
        return True
    # Nokta dizisi (içindekiler satırı)
    if re.search(r"\.{4,}", t):
        return True
    return False


def get_embedding_chunks(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    ADIM 2–3: Ham bloklardan RAG embedding adayı chunk'ları üretir.
    - is_noise=True veya TOC discard → atlanır
    - Sadece başlık (altında paragraf yok) → atlanır
    - Başlık + en az bir paragraf/list_item/table → tek chunk: title + content birlikte
    Çıktı: [ {"title": "...", "content": "...", "page", "section_id", "chunk_id", "block_ids"}, ... ]
    """
    blocks = result.get("blocks") or []
    content_types = ("paragraph", "list_item", "table")
    chunks: list[dict[str, Any]] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.get("is_noise") or _should_discard_toc(b):
            i += 1
            continue
        if b.get("type") == "title":
            title_text = (b.get("text") or "").strip()
            # Yazar/isim benzeri kısa başlık (tek satır, çok kısa) → atla
            if len(title_text) < 4 or (" " not in title_text and len(title_text) < 20):
                i += 1
                continue
            content_parts: list[str] = []
            block_ids: list[str] = [b.get("block_id", "")]
            j = i + 1
            while j < len(blocks):
                nb = blocks[j]
                if nb.get("type") == "title":
                    break
                if nb.get("is_noise") or _should_discard_toc(nb):
                    j += 1
                    continue
                if nb.get("type") in content_types:
                    content_parts.append((nb.get("text") or "").strip())
                    block_ids.append(nb.get("block_id", ""))
                j += 1
            if content_parts:
                chunks.append({
                    "title": title_text,
                    "content": "\n\n".join(content_parts),
                    "page": b.get("page"),
                    "section_id": b.get("section_id", ""),
                    "section_title": b.get("section_title", ""),
                    "chunk_id": b.get("chunk_id", ""),
                    "block_ids": block_ids,
                })
            i = j
            continue
        if b.get("type") in content_types and not (b.get("is_noise") or _should_discard_toc(b)):
            content_parts = [(b.get("text") or "").strip()]
            block_ids = [b.get("block_id", "")]
            j = i + 1
            while j < len(blocks):
                nb = blocks[j]
                if nb.get("type") == "title":
                    break
                if nb.get("is_noise") or _should_discard_toc(nb):
                    j += 1
                    continue
                if nb.get("type") in content_types:
                    content_parts.append((nb.get("text") or "").strip())
                    block_ids.append(nb.get("block_id", ""))
                j += 1
            chunks.append({
                "title": "",
                "content": "\n\n".join(content_parts),
                "page": b.get("page"),
                "section_id": b.get("section_id", ""),
                "section_title": b.get("section_title", ""),
                "chunk_id": b.get("chunk_id", ""),
                "block_ids": block_ids,
            })
            i = j
            continue
        i += 1
    return chunks


def _element_from_unstructured(obj: Any) -> dict[str, Any]:
    """Unstructured element objesini schema'ya uygun dict'e çevirir."""
    meta = getattr(obj, "metadata", None)
    meta_dict = meta.to_dict() if meta and hasattr(meta, "to_dict") else (meta if isinstance(meta, dict) else {})
    meta_dict = meta_dict or {}

    coords = meta_dict.get("coordinates")
    bbox = None
    if coords is not None:
        if hasattr(coords, "points") and coords.points:
            pts = coords.points
            if len(pts) >= 4:
                bbox = [pts[0][0], pts[0][1], pts[2][0], pts[2][1]]
        elif isinstance(coords, (list, tuple)) and len(coords) >= 4:
            bbox = list(coords)[:4]
        elif isinstance(coords, dict):
            bbox = coords

    category = getattr(obj, "category", None) or getattr(obj, "type", None) or "other"
    if isinstance(category, type):
        category = category.__name__ if hasattr(category, "__name__") else "other"
    text = getattr(obj, "text", None)
    if text is None:
        text = str(obj) if obj else ""

    out = {
        "category": str(category) if category else "other",
        "text": text if isinstance(text, str) else str(text),
        "page_number": meta_dict.get("page_number"),
    }
    if bbox is not None:
        out["bbox"] = bbox
    return out


def pdf_to_structure(pdf_path: str | Path) -> dict[str, Any]:
    """
    PDF dosyasını Unstructured ile parse edip yapısal JSON şemasına dönüştürür.
    Her zaman strategy="fast" kullanır (Tesseract / unstructured_inference gerekmez).
    Gereksinim: pip install unstructured[pdf]
    """
    try:
        from unstructured.partition.auto import partition
    except ImportError:
        raise ImportError(
            "PDF desteği için: pip install 'unstructured[pdf]'"
        ) from None

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    raw_elements = partition(filename=str(path), strategy="fast")
    if not raw_elements:
        return extract_structure([])
    elements = []
    for el in raw_elements:
        elements.append(_element_from_unstructured(el))
    return extract_structure(elements)


def _main() -> None:
    argv = sys.argv[1:]
    if not argv:
        sample = [
            {"category": "Title", "text": "Report 2024", "page_number": 1},
            {"category": "NarrativeText", "text": "This is a paragraph.", "page_number": 1, "bbox": [10, 20, 100, 30]},
            {"category": "ListItem", "text": "- Item one", "page_number": 2},
        ]
        print(to_json(sample))
        return

    pdf_path = argv[0]
    out_path = None
    if "-o" in argv:
        i = argv.index("-o")
        if i + 1 < len(argv):
            out_path = argv[i + 1]

    result = pdf_to_structure(pdf_path)
    if not result["blocks"]:
        print("Uyarı: Hiç blok çıkarılmadı. PDF taranmış (görüntü) olabilir; 'fast' modu OCR yapmaz.", file=sys.stderr)
    json_str = json.dumps(result, ensure_ascii=False, indent=2)

    if out_path:
        Path(out_path).write_text(json_str, encoding="utf-8")
        print(f"Yazıldı: {out_path}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    _main()
