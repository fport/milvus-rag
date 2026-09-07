# 3. Retrieval

Dört kanal, bir yönlendirici ve cevabın corpus'ta olmadığı durumda ne yapılacağına dair
bir karar. Buradaki her aşamanın bir sayısı var — ve ikisi beklenenin tam tersi çıktı.

!!! done "Bu aşamanın sahibi olduğu yerler"

    `search/routing.py` (regex yönlendirici), `search/retrieve.py` (kanallar, RRF,
    top-k, kanal başına skorlar), `search/rerank.py` (cross-encoder, kapalı) ve
    `config.py` içindeki üç bantlı abstain eşikleri.

## Yönlendirme: tek bir regex, bedava MRR

Bir embedding'in `handleAuthCallback` hakkında söyleyecek işe yarar hiçbir şeyi yoktur.
Bu bir kelime değildir, dağılımsal bir anlam taşımaz ve ona en yakın vektörler yalnızca
benzer görünen başka camelCase tanımlayıcılardır. BM25 onu birebir bulur, çünkü lexical
aramanın işi tam olarak budur.

Düz bir cümlede durum tam tersi.

```python
# search/routing.py
_SYMBOL_SHAPED = re.compile(r"^[A-Za-z_][\w.\-/:]*$")
_CAMEL_BOUNDARY = re.compile(r"[a-z0-9][A-Z]")

def looks_like_symbol(query: str) -> bool:
    """A single token with a boundary in it: camelCase, snake_case, kebab-case, a.b.c, a/b.

    Deliberately narrow: a false positive sends a real question to BM25 (measurably
    bad on prose); a false negative only gives up an improvement.
    """
```

O docstring'deki asimetri, tasarımın kendisi. Bir yönde yanılmak iyi bir cevaba mal
oluyor; diğer yönde yanılmak yalnızca bir iyileştirmeye. Bu yüzden desen tek bir token
*ve* içinde bir sınır istiyor — boşluk yok; ya camelCase, ya alt çizgi, ya nokta, ya
eğik çizgi, ya da tamamı büyük harf.

!!! measured "Neden her zaman iki kanal birleştirilmiyor"

    | Mod | Recall@8 | MRR | TR metin R@8 | p50 |
    |---|---|---|---|---|
    | yalnızca BM25 | 0.405 | 0.240 | 0.263 | 2 ms |
    | yalnızca dense | 0.786 | 0.678 | 0.684 | 32 ms |
    | hybrid (hep RRF) | 0.786 | 0.604 | 0.684 | 40 ms |
    | **auto** (sembol→BM25, metin→dense) ✓ | **0.786** | **0.690** | 0.684 | 34 ms |

    Her zaman birleştirmek, tek başına dense kanaldan *daha kötü* (MRR 0.604'e karşı
    0.678) ve sebebi mekanik: RRF, başarısız olan kanalın en iyi tahminini yukarı
    taşıyor. Düz metin bir soruda BM25'in ilk sonucu bir anahtar kelime tesadüfüdür ve
    birleştirme ona hak etmediği bir sıra ağırlığı verir.

    Yönlendirme bir regex'e mal oluyor ve yalnız dense kanala göre 0.012 MRR
    kazandırıyor. Bir LLM yönlendirici aynı şeyi bir ağ çağrısı karşılığında verirdi.

## RRF, kullanıldığı yerde

`RAG_PROSE_MODE=hybrid` iki listeyi Reciprocal Rank Fusion ile birleştiriyor:
`Σ 1/(k + rank)`, `RAG_RRF_K=60`.

Skorlar yerine sıraları kullanmasının sebebi, iki skorun aynı ölçekte olmaması —
kosinüs 0–1 arasında yaşıyor, BM25 kabaca 0–30 arasında. Ağırlıklı bir toplam
normalizasyon sabiti gerektirir ve o sabit corpus'la birlikte kayar: bir TypeScript
monoreposunda çalışan ağırlıklar bir Go servisinde yanlıştır. Sıranın birimi yoktur.

## Kapatılan cross-encoder

Plan sıradandı: 40 aday getir, `bge-reranker-v2-m3` her birini sorunun yanında okusun,
en iyi 8'i tut. Cross-encoder'lar sıralamada bi-encoder'ları güvenilir biçimde yener ve
recall payı gerçek — Recall@40 0.95, Recall@8 ise 0.786.

!!! measured "Sıralamayı kötüleştirdi"

    | Ayar | Recall@8 | MRR | TR metin R@8 | p50 |
    |---|---|---|---|---|
    | auto + dense | 0.786 | **0.690** | 0.684 | 34 ms |
    | hybrid + rerank | 0.762 | 0.508 | 0.579 | 4389 ms |
    | auto + rerank | 0.762 | **0.514** | 0.579 | 2050 ms |

    MRR dörtte bir düştü, p50 34 ms'den saniyelere çıktı. `RAG_RERANK_ENABLED`
    varsayılanı `false`.

k=8 ile k=40 arasındaki +0.17'lik recall boşluğu hâlâ orada, kapanmamış durumda.
Saklanmak yerine yazılıyor, çünkü bir sonraki iyileştirme için en bariz yer orası — ve
daha iyi bir reranker onu kapatırsa, bu satırın yerini almak yerine deftere yeni bir
satır olarak giriyor.

"Ölçülmüş" olmanın anlamı bu. Beklenen sonuç kazançtı; gözlenen sonuç kayıp oldu; bayrak
duruyor ki bir sonraki kişi iki sonuca da güvenmek yerine karşılaştırmayı kendi
corpus'unda yeniden çalıştırabilsin.

## Her sonuç kanal skorlarını taşıyor

```json
{
  "path": "src/queue/worker.ts",
  "symbol": "enqueueWebhookEvent",
  "start_line": 42,
  "end_line": 71,
  "category": "code",
  "scores": {"dense": 0.71, "bm25": 12.4, "rrf": 0.031}
}
```

Neyi hangi kanalın bulduğu yalnızca arayüzde değil, veri yapısının içinde. Bir
ablasyonu mümkün kılan tek şey bu: eval koşumu bir sonucu bir kanala yazmak için
bunları okuyor, bir hata ayıklama oturumu da yanlış bir chunk'ın neden başa geldiğini
açıklamak için.

## "Cevap yok" diyebilmek

Bir demoyu, bir ajanın güvenebileceği bir şeyden ayıran kısım burası.

kNN araması "en yakın k" demektir. "Yakında hiçbir şey yok" diye bir kavramı yoktur.
Retrieverye bir TypeScript API'sine karşı *"hiç beş yıldızlı bir tatil köyünde bulundun
mu?"* diye sor — kendinden emin biçimde sekiz chunk döndürür, en iyisi 0.366. Bunları
bir LLM'e ver, boş bir sonuçtan halüsinasyon imal etmiş olursun.

Akla ilk gelen çözüm — bir kosinüs eşiği — veriyle temas etmeye dayanmıyor:

!!! measured "Gri bölgede kosinüs ayırmıyor"

    Golden set top-1 dense: medyan 0.636, **min 0.526**.
    Alakasız top-1 dense: medyan 0.531, **maks 0.598**.

    Dağılımlar örtüşüyor. Tek bir eşik ya gerçek cevapları keser ya da çöpü içeri alır.

Bu yüzden bir kapı yerine üç bant — CRAG'in biçimi:

| Bant | Ayar | Davranış |
|---|---|---|
| < 0.45 | `RAG_MIN_DENSE_SCORE` | **düşürülür**, `dropped` sayılır |
| 0.45 – 0.55 | `RAG_WEAK_DENSE_SCORE` | döner, `weak_match: true` ile |
| ≥ 0.55 | — | normal |

Taban, golden set'teki en düşük gerçek cevabın epey altında duruyor, yani yalnızca saçma
kuyruğu temizliyor. Not ise bir **filtre değil, sinyal** — sonuçlar yine dönüyor ve
karar onları tüketene ait: ajana, LLM'e ya da arayüze bakan insana.

!!! measured "Neden not, neden reranker kapısı değil"

    | Kapı | abstain (13 negatif) | false_weak (42 pozitif) | eklenen gecikme |
    |---|---|---|---|
    | dense < 0.55 notu | 0.833 | 0.048 | 0 |
    | **taban 0.45 + not 0.55** ✓ | **0.846** | **0.048** | 0 |
    | rerank < 0.05 | 1.000 | **0.286** | +550 ms |
    | rerank < 0.5 | 1.000 | 0.595 | +550 ms |

    Reranker her negatifi yakalıyor — ve **üç sorgudan birinde** yanlış alarm veriyor.
    Üçte bir oranında güvenle görmezden gelebileceği bir uyarı gören bir ajan, onu her
    zaman görmezden gelmeyi öğrenir. Kosinüs notu %5 oranında yanlış tetikleniyor; bu,
    tutulmaya değer bir sinyal.

    0.45 tabanı golden set'ten hiçbir şey düşürmedi ve iki negatifi ("tatil köyü" 0.366,
    "Kafka rebalance" 0.418) sıfır sonuca indirdi.

### Eşikler evrensel değil

Bunlar **bu embedding modeli ve bu corpus için** ölçüldü. Kosinüs dağılımı modelle
birlikte kayıyor — aynı iş için Mistral'ın embedding'leri 0.73 civarına, Gemini'ninkiler
0.46 civarına düşüyor. Bu iki sayıyı başkasının yığını için sabit diye yayınlamak, tam
olarak bu projenin kaçınmaya çalıştığı türde ölçülmemiş bir iddia olurdu.

Bu yüzden eval raporu onları taşımak için gerekeni basıyor:

```json
"calibration": {
  "positive_min_top_dense": 0.498,
  "negative_max_top_dense": 0.587
}
```

Tabanı ilk sayının altına, notu ikisinin arasına koy. Embedding modeli her
değiştiğinde yeniden kalibre et.

## Bayraklar

Bunların her biri `POST /search` gövdesinde ve `rag eval` parametrelerinde istek başına
geçersiz kılınabiliyor — zincirin tamamı ablasyon için kuruldu.

| Bayrak | Varsayılan | Ne yapıyor |
|---|---|---|
| `RAG_SEARCH_MODE` | `auto` | sembol → BM25, düz metin → `RAG_PROSE_MODE` |
| `RAG_PROSE_MODE` | `dense` | `dense` ya da `hybrid` (dense + BM25 → RRF) |
| `RAG_RERANK_ENABLED` | `false` | 40 aday → cross-encoder → 8 — ölçüldü, zarar verdi |
| `RAG_TOP_K` | `8` | döndürülen sonuç sayısı |
| `RAG_CANDIDATES` | `40` | reranker'a verilen havuz |
| `RAG_MIN_DENSE_SCORE` | `0.45` | sert taban; 0 kapatır |
| `RAG_WEAK_DENSE_SCORE` | `0.55` | `weak_match` notunun eşiği |

```bash
uv run rag search "handleAuthCallback" --mode bm25
uv run rag search "webhook olayları nasıl kuyruğa alınıyor" -r my-api --json
```
