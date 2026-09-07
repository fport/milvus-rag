# 5. Ölçüm

Önceki dört sayfadaki her iddia bir yerde bir tablo satırı. Bu bir üslup tercihi değil —
bu tür sistemlerin davet ettiği başarısızlığa karşı tek savunma o: daha iyi *hissettiren*
bir şeyi yayınlamak.

!!! done "Bu aşamanın sahibi olduğu yerler"

    `eval.py` (golden set runner'ı), `evals/golden.example.jsonl` (şablon),
    `evals/results/<tarih>_<etiket>.json` (her koşu) ve README'deki ölçüm defteri.

## Tek kural

> Ölçülmemiş hiçbir şey yayınlanmaz. Her retrieval değişikliği tabloya "daha iyi
> hissettiriyor" olarak değil, bir satır olarak girer.

Bu projenin üç kararı, onları motive eden sezgiye **ters** çıktı: cross-encoder reranker
eklendi ve sonra kapatıldı; rerank tabanlı abstain kapısı en iyi negatif dedektörüydü
ve yine de kaybetti; en az önemli olması beklenen özellik olan zenginleştirme ise tek
seferde en büyük kazancı üretti. Bunların hiçbiri kodu okuyarak ya da elle birkaç sorgu
deneyerek görülemezdi.

## Golden set

Bir JSONL dosyası, satır başına bir soru, etiketler **repoyu okuyarak elle** yazılmış —
asla sistemi çalıştırıp döndürdüğünü kabul ederek değil. Sistemin kendi çıktısından
etiketlemek hiçbir şey ölçmez; sadece sistemin zaten yaptığını kayda geçirir.

```json
{"q": "where is the JWT issued?",
 "expect": ["src/auth/session.ts::createSession"],
 "mode": "any", "kind": "prose-en"}
```

| Alan | Anlamı |
|---|---|
| `expect` | `path` ya da `path::symbol`. **Asla satır numarası** — kod değişince satırlar kayar, semboller kaymaz |
| `mode` | `any` → bir girdi yeter; `all` (varsayılan) → hepsi dönmeli |
| `kind` | kırılım etiketi (`prose-en` / `prose-tr` / `symbol` / `negative`) — rapor buna göre ayrılıyor |

Tabloyu tek bir sayı olmaktan çıkarıp tanı koyucu yapan şey `kind` kırılımı. 0.786'lık
toplu bir Recall@8, sembol sorgularının 1.0 ve Türkçe düz metnin 0.684 aldığını gizler —
ve hangi düğmeyi çevireceğini söyleyen şey ortalama değil, o ayrım.

### Negatif vakalar kümenin parçası

```json
{"q": "hiç beş yıldızlı bir tatil köyünde bulundun mu?", "expect": []}
```

kNN araması "en yakın k" demektir. *Yakında hiçbir şey yok* diye bir kavramı yoktur: o
soru için de retriever sekiz chunk döndürüyor, en iyisi 0.366. Halüsinasyon tam olarak
orada başlıyor ve yalnızca cevaplanabilir sorulardan oluşan bir golden set bunu göremez.

`abstain_rate` (sistem bir negatifte ne sıklıkla "cevap yok" dedi), `false_weak_rate`
(aynı sinyal bir pozitifte ne sıklıkla yanlış tetiklendi) ile yan yana raporlanıyor. İki
sayı tek başına anlamsız: her şeye abstain diyen bir kapı, kusursuz bir abstain oranı
alır.

!!! measured "Hangi abstain kapısı kaldı"

    | Kapı | abstain (negatifler) | false_weak (42 pozitif) | eklenen gecikme |
    |---|---|---|---|
    | dense < 0.55 notu | 10/12 = 0.833 | 2/42 = 0.048 | 0 |
    | **taban 0.45 + not 0.55** ✓ | 11/13 = 0.846 | 2/42 = 0.048 | 0 |
    | rerank < 0.05 | 12/12 | 12/42 = 0.286 | +550 ms p50 |
    | rerank < 0.5 | 12/12 | 25/42 | +550 ms |

    Reranker her negatifi yakalıyor — ve gerçek cevapların p25'ine 0.039 veriyor. Üç
    sorgudan birinde yanlış alarm, bir ajanın notu tamamen görmezden gelmeyi öğrenmesine
    yeter; o noktada kusursuz dedektör hiçbir şey tespit etmiyor demektir. Kosinüs notu
    negatiflerde daha kötü ve pozitiflerde %5 yanılıyor — yayınlanan o oldu.

## Koşucu

```bash
G=evals/golden.example.jsonl
uv run rag eval $G -r my-api --tag dense  --mode dense  --no-rerank
uv run rag eval $G -r my-api --tag bm25   --mode bm25   --no-rerank
uv run rag eval $G -r my-api --tag hybrid --mode hybrid --no-rerank
uv run rag eval $G -r my-api --tag rerank --mode auto   --rerank
```

Her koşu `evals/results/<tarih>_<etiket>.json` dosyasını raporla *ve vaka başına her
satırla* yazıyor — soru, recall, karşılıklı sıra, gecikme, ne döndüğü, `top_dense`.
Satırları saklamak, şaşırtıcı bir toplu sonucun aylar sonra ona sebep olan iki soruya
kadar izlenebilmesini sağlıyor.

Repoya özgü golden set dosyaları gitignore'da. Yalnızca şablon yayınlanıyor: bir golden set,
senin iç dosya yollarının listesidir.

## Kalibrasyon modelle birlikte gezer

Bu projedeki eşikler (`0.45` taban, `0.55` not) evrensel sabitler değil. **Bu corpus'ta
BGE-M3 için** ölçüldüler. Aynı iş Mistral'ın embedding'lerinde 0.73 civarına, Gemini'de
0.46 civarına düşüyor. Bir sayıyı modeller arasında kopyalamak, ya her şeyi geçiren ya da
her şeyi engelleyen bir kapı üretir.

Bu yüzden rapor bir `calibration` bloğu taşıyor:

```json
"calibration": {"positive_min_top_dense": 0.498, "negative_max_top_dense": 0.587}
```

Bulunan pozitifler arasındaki en düşük top-dense ve negatifler arasındaki en yüksek.
Tarif mekanik: **taban birincinin altına, not ikisinin arasına.** Embedding modelini
değiştir, eval'i yeniden çalıştır, bloğu oku, iki sayıyı taşı.

O iki değerin örtüşmesi, kapının bir karar değil bir *not* olmasının dürüst sebebi de.
Pozitifler 0.498'de diplerken negatifler 0.587'ye çıkıyor — dağılımlar kesişiyor. Tek
bir kosinüs eşiği onları ayırmıyor, bu yüzden sistem gördüğünü bildiriyor ve kararı
tüketen veriyor.

## Defter

Tam tablo README'de duruyor; iki dilde tutuluyor, tarihli, corpus ve soru dağılımı
belirtilmiş hâlde. Burada kısaca:

!!! measured "Retrieval modları — 318 dosyalık TypeScript monorepo, 42 soru, k=8"

    | Etiket | Ayar | Recall@8 | MRR | TR metin R@8 | p50 |
    |---|---|---|---|---|---|
    | bm25 | yalnızca BM25 | 0.405 | 0.240 | 0.263 | 2 ms |
    | hybrid | dense+BM25 → RRF | 0.786 | 0.604 | 0.684 | 40 ms |
    | dense | yalnızca dense | 0.786 | 0.678 | 0.684 | 32 ms |
    | **auto+dense** ✓ | sembol→bm25, metin→dense | **0.786** | **0.690** | 0.684 | 34 ms |
    | hybrid k=40 | aday havuzu | 0.952 | — | 0.895 | 39 ms |
    | hybrid+rerank | 40→8, bge-reranker-v2-m3 | 0.762 | 0.508 | 0.579 | 4389 ms |

    Recall@40 = 0.952. Aday havuzunda bir reranker'ın kapatabileceği gerçek bir +0.17
    duruyor — denenen ise onu kapatmak yerine sıralamayı bozdu ve saniyelere mal oldu.
    Boşluk, giderilmiş bir kusur olarak değil, açık bir madde olarak kayıtta.

!!! measured "Chunking — AST'ye karşı düz pencere, aynı corpus iki kez indekslendi"

    | Etiket | chunk | Recall@8 | MRR | TR metin MRR | sembol MRR |
    |---|---|---|---|---|---|
    | **ast** ✓ | 2761 | **0.786** | **0.690** | 0.570 | **1.000** |
    | plain | 2100 | 0.762 | 0.598 | 0.518 | 0.321 |

    Recall bir soru kadar oynuyor. MRR 0.09 oynuyor ve neredeyse tamamı sembol
    sorguları. AST chunking daha fazlasını bulmuyor — doğru chunk'ı en üste koyuyor ve
    adını biliyor. Gramersiz bir dilin kaybettiği şeyin boyutu da bu: sıralama ve atıf,
    recall değil.

!!! measured "Zenginleştirme — 46 dosya / 423 chunk, iki kez indekslendi"

    | Etiket | Recall@8 | MRR | TR metin R@8 / MRR | false_weak |
    |---|---|---|---|---|
    | subset-plain | 0.929 | 0.839 | 0.895 / 0.778 | 0.119 |
    | **subset-enriched** | **1.000** | **0.912** | **1.000 / 0.932** | **0.048** |

    Projedeki tek seferlik en büyük kazanç — hem de **kapalı** yayınlanan özellikten.
    İngilizce dışı düz metinde +0.15 MRR, `false_weak` yarıya indi, abstain
    kımıldamadı: açıklamalar gerçek cevaplardaki güveni yükseltti ama alakasız olanlar
    için güven imal etmedi. Kapalı kalmasının sebebi varsayılan kurulumun LLM bütçesinin
    olmaması: yerel bir 9B'de dakikada ~11 açıklama, 2.8k chunk'lık bir repo için ~4 saat
    demek. Sonrası artımlı, chunk özeti + modele göre önbellekli.

## Defter ne işe yarıyor

Üç şey, hiçbiri süs değil:

1. **Geri dönüşü ucuzlatıyor.** Reranker hâlâ kod tabanında, `RAG_RERANK_ENABLED`
   arkasında, neden kapalı olduğunu söyleyen satırıyla birlikte. Daha iyi bir reranker
   aynı 42 soruya karşı yeni bir satır olarak giriyor — bir tartışma olarak değil.
2. **Beklentinin boyutunu belirliyor.** "AST chunking +0.09 MRR eder, çoğu sembolde"
   kullanılabilir bir cümle. "AST chunking daha iyidir" değil.
3. **Sürprizleri kayda geçiriyor.** Zenginleştirme, üzerinde en az durulan özellikti ve
   kazandı. Bu ancak yargılanmadan önce ölçüldüğü için bilinebilir hâle geldi.
