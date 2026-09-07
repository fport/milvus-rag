# 2. basamak — Birim olan parçalar

1. basamak korpusu her 1000 karakterde bir kesti. Sınır keyfîydi, dolayısıyla parçaların
yarısı bir şeyin yarısıydı.

2. basamak sınırı anlamlı kılıyor ve her parçaya bir ad veriyor.

## İki değişiklik

**Sınır karakter sayısından değil, yapıdan geliyor.** Kod için bu bir ayrıştırıcı;
düzyazı için önce başlıklar, sonra paragraflar.

**Her parça ne olduğunu taşıyor.** Dosya, sembol, üst öge, dil, tür. O üstveri bir sonucu
takip edilebilir bir atıfa çeviren şey — ve aynı zamanda erişimcinin eşleştirebileceği
fazladan metin.

İkinci değişiklik atlananı, ve ikisinin ucuz olanı.

## Embed ettiğin metni gösterdiğinden ayır

Bu basamaktaki en işe yarar tek ayrıntı:

```python
# ChunkRecord: `text` is the clean content that goes to the LLM;
# `header` is the "what am I, where do I live" lines prepended for
# embedding and BM25.
```

Tek parçadan iki alan:

| Alan | İçeriği | Kullanıldığı yer |
|---|---|---|
| `text` | yalnızca kod | atıf, LLM prompt'u |
| `indexed_text` | `header` + `text` | embedding, BM25 indeksi |

Başlık şuna benzer bir şey:

```text
src/queue/worker.ts · class QueueWorker · method drain
imports: redis, pino
```

O başlığı `text`'in içine kat, insana gösterdiğin her atıf üç satır üstveriyle başlasın.
`indexed_text`'ten çıkar, `handle` adlı bir metot repodaki diğer kırk `handle`'dan ayırt
edilemez olsun.

Bu basamaktan tek bir şey alacaksan şunu al: **gösterdiğinden fazlasını embed et.**
Hiçbir maliyeti yok ve bir parçanın, içinde yaşadığı sınıfın adıyla bulunabilmesinin
sebebi bu.

## Kod için: ayrıştırıcı, parantez sayma değil

```python
CODE_WITHOUT_GRAMMAR: dict[str, str] = {".sql": "sql", ".cob": "cobol", ".cbl": "cobol"}
```

tree-sitter, `{` ve `}` üzerinde bir regex değil — bir string literal içindeki parantez,
onu içeren ilk dosyada parantez saymayı bozar.

Bu projenin oturduğu dört kural:

1. Büyük bir sınıf üyelerine bölünüyor; başlık ve alanlar kendi parçası oluyor.
2. `RAG_CHUNK_MIN_BYTES` altındaki her şey bir komşusuyla birleşiyor — tek satırlık bir
   tip ya da bir import bloğu erişilebilir bir birim değil.
3. Hiçbir şey `RAG_CHUNK_MAX_BYTES`'ı (2000 B) aşmıyor. BGE-M3 8192 token kabul ediyor ve
   tuzak bu: uzun bir parçanın embedding'i bir **ortalamadır** ve ortalama hiçbir şeyi iyi
   temsil etmez. Tavanı aşan bir fonksiyon, sembol adını koruyan satır pencerelerine
   bölünüyor.
4. Grameri, dil başına bir tablo tutmak yerine dosya uzantısından türet.
   `tree-sitter-language-pack`'te 371 gramer var; tablo, 372'nci konusunda yanılmak
   demekti.

Ayrıntılar ve dil kapsamı kuralı: **[İndeksleme](../02-indexing.md)**.

## Düzyazı için: çok daha az iş

Bu basamağa çıkmak için ayrıştırıcıya ihtiyacın yok.

- **Markdown**: başlıklardan böl, başlık izini (`Kılavuz › Faturalama › İadeler`) header
  olarak sakla. Çoğu korpus için faydanın çoğu bu.
- **Düz metin**: paragraflar, asgari boyuta kadar birleştirilmiş.
- **HTML**: `<h1>`–`<h3>` üzerinde markdown'la aynısı.
- **PDF**: asıl iş burada ve o iş parçalama değil, çıkarım. Önce yerleşimi doğru al; kötü
  bir çıkarımı parçalamak gürültüyü cilalamaktır.

## Ne kazandırıyor

!!! measured "AST parçalamaya karşı düz pencereler, aynı korpus iki kez indekslendi"

    | Parçalayıcı | parça | Recall@8 | MRR | sembol MRR | İngilizce dışı metin MRR |
    |---|---|---|---|---|---|
    | **AST** ✓ | 2761 | **0.786** | **0.690** | **1.000** | 0.570 |
    | düz pencere | 2100 | 0.762 | 0.598 | 0.321 | 0.518 |

    Recall bir soru oynadı. MRR 0.09 oynadı ve neredeyse tamamı sembol sorguları:
    **0.321 → 1.000**.

Bunu dürüstçe oku: yapısal parçalama **daha fazlasını bulmuyor**. Doğru parçayı en üste
koyuyor ve adını biliyor. Recall sütunu neredeyse kımıldamıyor.

Beklentinin doğru boyutu da bu. Sorunun "cevap hiç gelmiyor" ise 2. basamak senin basamağın
değil. Sorunun "6. sırada geliyor ve atıf işe yaramıyor" ise, o.

## Neyi düzeltmiyor

`iter_source_files` hâlâ adıyla bulunamıyor — sembolü iliştirilmiş bir parça da bir
embedding ile getiriliyor ve embedding hâlâ kimlik yapamıyor.

O, **[3. basamak](03-hybrid.md)**.
