# 2. İndeksleme

Bir repoyu aranabilir bir şeye çevirmek. Dört adım — ve ilki, diğer üçünün ne kadar iyi
olabileceğini belirliyor.

!!! done "Bu aşamanın sahibi olduğu yerler"

    `index/chunk.py` (tree-sitter), `index/scrub.py` (sırlar ve kişisel veri),
    `index/embed.py` (BGE-M3), `index/enrich.py` (isteğe bağlı açıklamalar) ve
    `index/store.py` (tek Milvus koleksiyonu).

## Bir parça, bir kod birimidir

Çoğu RAG yığınının varsayılan parçalayıcısı bir karakter penceresidir. Düz metinde bu
savunulabilir. Kodda bir fonksiyonu ortasından keser — ve yarım bir fonksiyon, kimsenin
bir soruyu cevaplayabileceği bir şey değildir.

Bu yüzden sınırlar gerçek bir ayrıştırıcıdan geliyor: tree-sitter, süslü parantez
sayarak değil — çünkü bir string literal içindeki `}`, bunu içeren ilk dosyada parantez
saymayı bozar.

Dört kural:

1. **Büyük bir sınıf ya da namespace üyelerine bölünüyor.** Başlık ve alanlar kendi
   parçası oluyor.
2. **Küçük şeyler bir komşusuyla birleştiriliyor.** Tek satırlık bir tip, kısa bir
   const, bir import bloğu, bir JSDoc yorumu — hiçbir şey `RAG_CHUNK_MIN_BYTES`'ın
   altına inmiyor.
3. **Hiçbir parça `RAG_CHUNK_MAX_BYTES`'ı aşmıyor** (2000 B, kabaca 500 token). BGE-M3
   8192 token kabul ediyor ve tuzak tam olarak bu: uzun bir parçanın embedding'i bir
   *ortalamaya* dönüşür ve hiçbir şeyi iyi temsil etmez. Tek başına tavanı aşan bir
   fonksiyon, sembol adını koruyan satır pencerelerine bölünüyor.
4. **`text` temiz kalıyor.** LLM'e ve atıfa giden şey o. "Ben neyim, nerede
   yaşıyorum" başlığı — dosya, sembol, üst sınıf, import'lar — yalnızca embed'lenen ve
   BM25 ile indekslenen alana, `indexed_text`'e giriyor.

Yanlış yapılması kolay olan kural 4. Başlığı `text`'e koymak her atıfı üç satır
üstveriyle başlatır; `indexed_text`'ten çıkarmak ise `handle` adlı bir metodu diğer kırk
tanesinden ayırt edilemez kılar.

### Dil kapsamı bir tablo değil, bir kuraldır

```python
# The extension name is the grammar name for most languages: .lua, .vue, .razor, .zig.
# The ones that differ (.ts, .cs, .kt) come from a small alias table validated against
# the pack. Anything without a grammar is still indexed, as plain windows.
CODE_WITHOUT_GRAMMAR: dict[str, str] = {".sql": "sql", ".cob": "cobol", ".cbl": "cobol"}
```

`tree-sitter-language-pack` 371 gramer getiriyor. Dil başına kural tablosu yazmak, 371
girdiyi bakımda tutup 372'ncisi konusunda yanılmak demekti; grameri uzantıdan türetmek
ise yeni bir dilin, paket onu desteklediği gün çalışması demek.

İki gramer adıyla dışlanmış durumda, çünkü ayrıştırıcıyı çökertiyor ya da kilitliyorlar:
`sql` ve `cobol`. Yine de satır pencereleriyle indeksleniyorlar.

!!! measured "AST parçalama gerçekte ne kazandırıyor"

    Aynı korpus iki kez indekslendi — bir kez tree-sitter'la, bir kez düz pencerelerle;
    aynı altın küme, k=8:

    | Parçalayıcı | parça | Recall@8 | MRR | sembol R@8 / MRR | TR metin MRR |
    |---|---|---|---|---|---|
    | **AST** ✓ | 2761 | **0.786** | **0.690** | 1.0 / **1.0** | 0.570 |
    | düz pencere | 2100 | 0.762 | 0.598 | 0.75 / 0.321 | 0.518 |

    Recall bir soru farkediyor. MRR 0.09 fark ediyor ve **neredeyse tamamı sembol
    sorgularında** (MRR 0.321 → 1.0). İngilizce düz metinde düz pencere başabaş, hatta
    bir soru önde.

    AST parçalama daha fazlasını bulmuyor. Doğru parçayı en üste koyuyor ve adını
    söylüyor. Gramersiz kalan bir dilde kaybedilen şey sıralama ve atıf kalitesi, recall
    değil — eksik bir gramerin acil durum olmamasının sebebi de bu.

## Saklamadan önce temizlik

Bir vektör geri çevrilemez. Yanında saklanan `content` alanı çevrilebilir — her atıfta
birebir dönüyor ve oradan önbelleğe, loglara ve modelin cevaplarına yolculuk ediyor. Bu
yüzden temizlik, indekse hiçbir şey ulaşmadan önce yapılıyor.

Kurallar kasıtlı olarak muhafazakâr ve bu kelimenin arkasında bir sayı var:

!!! measured "Muhafazakâr, kapsamlıyı yener"

    Gerçek bir repoda, *adlara* göre karartan bir kural **78 dosyada 247 karartma**
    üretti ve neredeyse hiçbiri sır değildi: `token: text(` bir veritabanı kolonu,
    `secret: string` bir tip bildirimi.

    Yayınlanan kurallar aynı repoda **2 karartma** yaptı. Biri gerçek bir jetondu.

Desenler adları değil biçimleri yakalıyor — `sk-`, `ghp_`, `xox[baprs]-`, `AKIA…`,
JWT'ler, e-postalar, telefon numaraları — artı yalnızca değer atanmış BÜYÜK_YILAN
biçimli bir anahtarda tetiklenen tek bir ad tabanlı kural:

```python
# DB_PASSWORD=hunter2 is a secret; `promptTokens` and `this.accessToken` are not.
_ASSIGNED_SECRET = re.compile(
    r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|APIKEY|API_KEY|PAT)[A-Z0-9_]*"
    r"|(?:PASSWORD|SECRET|TOKEN|APIKEY|API_KEY|PRIVATE_KEY)[A-Z0-9_]*)"
    r"(\s*[:=]\s*)"
    r"(\"[^\"\n]*\"|\'[^\'\n]*\'|[^\s\"\'#,;]+)"
)
```

…ve atama gibi görünüp atama olmayanlar için bir dışlama listesi: `process.env.X`,
`${…}`, bir tip adı, bir fonksiyon çağrısı, `this.…`, bir URL şeması ve yer tutucu
sözlüğü (`your-`, `change-me`, `xxx`, `example`).

Ayrılmış e-posta alan adlarına (`example.com`, `*.test`, `*.invalid`, `localhost`)
dokunulmuyor, çünkü bir doküman yorumundaki örnek adresi karartmak bilgi siler ve
kimseyi korumaz.

## Embedding

BGE-M3, 1024 boyut, varsayılan olarak yerelde çalışıyor. İki özelliği önemli.

**Çok dilli** — bir dildeki sorunun, başka bir dilde yorumlanmış kodu bulabilmesinin
tüm sebebi bu. MiniLM ile yapılan önceki deney kontrol grubu: Türkçe düz metin
**Recall@5 = 0.04** aldı. Aynı sorular BGE-M3'te 0.684 alıyor. Bu bir ayar farkı değil,
farklı bir yetenek.

**Makinede çalışıyor**, yani kod hiç dışarı çıkmıyor. Bu takası tercih edenler için
`RAG_EMBEDDING_BACKEND=openai` barındırılan bir modele geçiyor.

```python
# index/embed.py
"""Vectors are L2-normalized on the way out and searched with COSINE — mixing metrics
silently produces wrong ordering, so it is done in one place."""
```

Normalizasyonun tek bir yerde olması titizlik değil. İç çarpımla aranan normalize bir
vektör ile kosinüsle aranan normalize edilmemiş bir vektör, *makul görünen* ama ince
biçimde yanlış sıralamalar üretir — ve hiçbir yerde hata çıkmaz.

Model süreç başına bir kez yükleniyor, istek başına değil — ilk yükleme 10–20 saniye.
Torch fonksiyonların içinde import ediliyor, böylece CLI'daki `import milvus_rag` bir
yardım mesajı basmak için derin öğrenme yığını çekmiyor.

## Zenginleştirme, varsayılanı kapalı

Kod neredeyse hiç doğal dil içermez; içerdiği kadarı da yazarının yorumları hangi dilde
yazdıysa o dildedir. Çare daha iyi bir eşleştirici değil; eksik olan düzyazıyı yazmak.
Bu, Anthropic'in bağlamsal erişim fikri: LLM'e dosyanın tamamını verip her parça için
iki üç cümle istemek, sonra bunları da yanına embed'lemek.

!!! measured "Zenginleştirme ne kazandırıyor, neye mal oluyor"

    46 dosya / 423 parça, aynı korpus iki kez indekslendi, açıklamalar yerel bir
    `qwen3.5:9b`'den:

    | Kol | Recall@8 | MRR | TR metin R@8 / MRR | false_weak |
    |---|---|---|---|---|
    | düz | 0.929 | 0.839 | 0.895 / 0.778 | 0.119 |
    | **zenginleştirilmiş** | **1.000** | **0.912** | **1.000 / 0.932** | **0.048** |

    Kazanç tam olarak öngörülen yere düşüyor: **Türkçe düz metinde +0.15 MRR**,
    İngilizcede +0.01. `false_weak` yarıya indi, çünkü gerçek cevapların yoğun skorları
    yükseliyor ve 0.55 notu daha az yanlış tetikleniyor. Negatiflerde çekimserlik
    kımıldamadı — açıklamalar alakasız bir soru için güven imal etmedi.

    Maliyet: yerel bir 9B'de dakikada ~11 açıklama, yani 2.8k parçalık bir repo ilk
    seferde yaklaşık dört saat. Sonrası artımlı (parça özeti + modele göre önbellekli),
    bulut modeliyle dakikalar.

Varsayılanı **kapalı**, çünkü anahtarsız bir kurulumda her `add-*` saatlerce Ollama
döndürürdü. Soru trafiğin İngilizce değilse aç ve `RAG_ENRICH_LANGUAGE`'ı ona göre ayarla
— ölçülen kazanç, insanların gerçekten sorduğu dildeki düzyazıdan geliyor.

İki koruma, ikisi de bir 7B modelin başarısızlığını izlemekten çıktı:

- **Kapsam kontrolü.** Küçük bir model seve seve tek bir nesne döndürüp gerisini sessizce
  düşürür. Cevap, parça listesine karşı kontrol ediliyor.
- **Alfabe kontrolü.** `qwen2.5:7b` yük altında cümle ortasında Çinceye kayıyor ve bunu
  söylemiyor. Yanlış alfabedeki bir açıklama reddediliyor.

## Tek koleksiyon, bölüm anahtarı `repo_id`

```python
# index/store.py
"""Dense and sparse live in the same collection. Nobody writes the sparse field;
Milvus' own BM25 Function derives it from `indexed_text`, so there is no separate
lexical index."""
```

Milvus'un alternatiflerine tercih edilme sebebi o son cümle. Hibrit arama normalde iki
indeksi ayakta tutup tutarlı tutmak demektir; burada BM25 aynı koleksiyondaki bir alan
üzerinde bir fonksiyon, yani bir yazma sadece bir yazma.

Bölüm anahtarı sayesinde bir `repo_id in [...]` filtresi taramak ve filtrelemek yerine
yalnızca ilgili bölümlere iniyor.

Silme ve güncelleme `repo_id + path` üzerinden gidiyor. Arşivleme ve yumuşak silme yok:
Milvus'taki şey türetilmiş veri, doğruluk kaynağı repo. Repodan yeniden üretilemeyen
hiçbir şeyin orada işi yok.

```bash
# Dizinden aranabilir koleksiyona, indeksleme yolunun tamamı
uv run rag add-local ~/code/my-api --name my-api
uv run rag sync my-api --force     # her şeyi yeniden parçala ve yeniden embed'le
```
