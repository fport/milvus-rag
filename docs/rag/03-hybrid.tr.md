# 3. basamak — Sözcüksel arama ve yönlendirme

Bir embedding "bu neyle ilgili" sorusunu cevaplar. Gerçek sorguların çok büyük bir kısmı
hiçbir şeyle ilgili değildir — bir isimdir.

```text
handleAuthCallback        ERR_CONN_RESET        RAG_WEAK_DENSE_SCORE
UserNotFoundException     invoice_line_items    --no-verify
```

Bunlar için benzerlik yanlış soru. En yakın şeyi değil, *o* şeyi istiyorsun ve bunun
standart aracı embedding'den otuz yıl daha eski: **BM25**, nadirliğe göre ağırlıklı
birebir terim eşleşmesi.

## İki kanal

Vektör indeksinin yanına bir sözcüksel indeks ekle. Her parça ikisine birden giriyor.

| Kanal | İyi olduğu şey | Kör olduğu şey |
|---|---|---|
| yoğun (embedding) | başka sözcüklerle ifade, diller arası, "bu neyle ilgili" | birebir kimlik, nadir jetonlar |
| BM25 (sözcüksel) | tanımlayıcılar, hata kodları, yapılandırma anahtarları, tırnak içi metinler | eş anlamlılar, başka bir dil, yeniden ifade edilmiş her şey |

Bu projede ikisi de **tek bir Milvus koleksiyonunda**: seyrek alanı kimse yazmıyor,
Milvus'un kendi BM25 Function'ı onu `indexed_text`'ten türetiyor. Bu, basamağın alışılmış
maliyetini — tutarlı tutulması gereken iki indeks — ortadan kaldırıyor ve Milvus'un burada
olmasının büyük sebebi. Deposun bunu yapamıyorsa bu basamak, ikinci bir indeks çalıştırmak
ve ikisine de yazmak demek.

## Sonra: varsayılan olarak birleştirme

İnsanları şaşırtan kısım burası ve ölçüldü.

Akla ilk gelen tasarım "her zaman ikisini de koştur, sonuçları kaynaştır". Alışılmış
kaynaştırıcı Reciprocal Rank Fusion: `Σ 1/(k + rank)`; skorlar yerine sıralar, çünkü
kosinüs 0–1 arasında, BM25 kabaca 0–30 arasında yaşıyor ve ikisinin ağırlıklı toplamı,
korpusla birlikte kayan bir normalizasyon sabiti gerektiriyor.

!!! measured "Her zaman hybrid, tek başına yoğundan daha kötü"

    | Mod | Recall@8 | MRR | İngilizce dışı R@8 | p50 |
    |---|---|---|---|---|
    | yalnızca BM25 | 0.405 | 0.240 | 0.263 | 2 ms |
    | yalnızca yoğun | 0.786 | 0.678 | 0.684 | 32 ms |
    | hybrid (hep RRF) | 0.786 | **0.604** | 0.684 | 40 ms |
    | **auto** (biçime göre yönlendir) ✓ | **0.786** | **0.690** | 0.684 | 34 ms |

    Her zaman kaynaştırmak, yalnızca yoğun kanalı kullanmaya karşı **0.074 MRR'a mal oldu**.

Mekanizma bir kez görülünce şaşırtıcı değil: düz metin bir soruda BM25'in ilk sonucu bir
anahtar kelime tesadüfüdür. RRF bunu bilmez. Başarısız olan kanalın en iyi tahminini yukarı
taşır ve o tahmin gerçek bir cevabın yerini alır.

Kaynaştırma, iki kanalın da sorguya adil bir şans bulduğunu varsayar. Yalnızca birinin
hizmet edebileceği bir sorguda etkin biçimde zararlıdır.

## Onun yerine yönlendir

Sorgunun hangi kanala ait olduğuna karar ver ve o kanalı kullan.

```python
_SYMBOL_SHAPED = re.compile(r"^[A-Za-z_][\w.\-/:]*$")
_CAMEL_BOUNDARY = re.compile(r"[a-z0-9][A-Z]")

def looks_like_symbol(query: str) -> bool:
    """A single token with a boundary in it: camelCase, snake_case, kebab-case, a.b.c, a/b.

    Deliberately narrow: a false positive sends a real question to BM25 (measurably
    bad on prose); a false negative only gives up an improvement.
    """
```

O docstring tasarımın kendisi. İki hata simetrik değil:

- **yanlış pozitif** (gerçek bir soru BM25'e yönlendirilir) → kötü bir cevap
- **yanlış negatif** (bir sembol yoğun kanala yönlendirilir) → kaçırılmış bir iyileştirme

Bu yüzden desen kasıtlı olarak dar: tek jeton, boşluk yok, içeride bir sınır. Gerisi düz
metin.

Bir regex, ve yalnız yoğun kanala karşı 0.012 MRR kazandırırken sembol sorgularını kusursuz
yapıyor (BM25 onlarda 1.0/1.0 alıyor). Bir LLM yönlendirici aynı şeyi bir ağ çağrısı ve bir
belirsizlik kaynağı karşılığında verirdi.

!!! done "Hybrid'in doğru varsayılan *olduğu* yer"

    Yönlendirme burada işe yarıyor, çünkü kod sorguları temiz biçimde iki şekle ayrılıyor.
    Ayrılmadığı bir korpusta — içinde ürün adları, SKU'lar ya da parça numaraları da geçen
    karışık doğal dil sorguları — temiz bir kural yok ve dürüst varsayılan her zaman RRF.

    Basamak "iki kanalı da bulundur ve bilinçli kullan". `RAG_PROSE_MODE=hybrid` burada
    kaynaştırmayı geri açıyor; bu tabloyu devralmak yerine kendi korpusunda ölç.

## Neyi düzeltmiyor

Erişimci artık iki sorgu biçimi için de doğru şeyi buluyor. Aynı güvenle, beş yıldızlı bir
tatil köyü sorusuna da sekiz parça döndürecek.

O, **[4. basamak](04-honesty.md)**.
