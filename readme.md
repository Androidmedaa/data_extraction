pip install "unstructured[pdf]"

(Tesseract / hi_res isteğe bağlı; daha iyi layout için sonradan eklenebilir.)

---

## Ham çıktı alanları

| Alan | Açıklama |
|------|----------|
| section_id | En son başlıktan slug (örn. kullanilan_teknolojiler). |
| section_title | En son başlığın metni. |
| hierarchy_level | 1=başlık, 2=list_item, 3=paragraf/tablo, 4=header/footer. |
| chunk_id | `page_{sayfa}_section_{section_id}_{sıra}` — embedding/retrieval/citation. |
| block_index_in_page | Sayfa içi sıra. |
| is_noise | true → RAG'de embed edilmemeli. |
| normalized_text | Birleştirilmiş boşluk, NFC, trim. |

**Not:** Bu çıktı ham ingestion çıktısıdır; embedding öncesi filtreleme gerektirir.

---

## Buradan sonra NE YAPMAN gerekiyor? (net adımlar)

### ADIM 1 – İçindekiler'i çöpe at
- **Kural:** `type=other` + (metinde `.....` / sayfa numarası / "İçindekiler") → **discard**
- Kod: `is_noise` + `get_embedding_chunks()` içinde `_should_discard_toc()` ile atılır.

### ADIM 2 – Gerçek içerik sayfalarına bak
- **Soru:** "Bu başlığın ALTINDA açıklayıcı paragraf var mı?"
- Sadece başlık → embedding'e **alma**
- Başlık + açıklama → **embedding adayı**
- Kod: `get_embedding_chunks()` sadece "başlık + en az bir paragraf/list/table" olan bölümleri chunk yapar.

### ADIM 3 – Title + Paragraph birlikte ele al
- **İdeal embedding birimi:** Başlık + içerik tek chunk.
- Sadece başlık ❌ / Sadece paragraf ❌ / **Birlikte** ✅
- Kod: `get_embedding_chunks(result)` → `{"title": "...", "content": "...", "page", "section_id", "chunk_id", "block_ids"}`

```python
from main import pdf_to_structure, get_embedding_chunks
result = pdf_to_structure("rapor.pdf")
chunks = get_embedding_chunks(result)   # Bunları embed et
```

---

## Mini karar tablosu (embedding?)

| Block türü | Embedding? |
|------------|------------|
| Title (konu başlığı) | ⚠️ Paragrafla birlikte |
| Title (isim / yazar) | ❌ |
| İçindekiler | ❌ |
| Sayfa numaralı başlık | ❌ |
| Açıklayıcı paragraf | ✅ |
| Kural / açıklama | ✅ |

---

## is_noise kuralları (embedding'e alınmaz)

| Tür | Kural |
|-----|--------|
| Tarih | Yaygın tarih formatları, kısa blok |
| Copyright | ©, "tüm hakları", "all rights reserved" |
| Link/URL | http, www, .com ağırlıklı |
| İçindekiler | Bölüm veya metin "içindekiler" |
| TOC satırları | "1. Giriş" tarzı |
| UI/ekran | "tıklayın", "butona", "ekran açılır" vb. kısa |
| Boilerplate | Aynı metin 3+ sayfada |
