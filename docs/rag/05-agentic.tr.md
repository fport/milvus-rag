# 5. basamak — Ajanlı erişim

1–4 basamakları bir hat kuruyor: soru girer, `k` parça çıkar, tek atış. O biçimin sert
bir tavanı var ve bu bir sıralama problemi değil.

> *"Hangi uçlara kimlik doğrulaması olmadan erişilebiliyor?"*

Hiçbir `k` parça bunu cevaplamaz. Auth middleware'ini bulman, nereye takıldığını bulman,
sonra onun altında olmayan rotaları saymanız gerekir. Üç arama — ve **birincinin cevabını
almadan ikinci sorguyu yazamazsın**.

5. basamak bunu düzelten kayma: erişim bir hat aşaması olmayı bırakıp **başka bir şeyin
döngü içinde çağırdığı bir araca** dönüşüyor.

## Gerçekte ne değişiyor

| | 1–4 basamakları | 5. basamak |
|---|---|---|
| Sorguyu kim yazıyor | senin kodun, bir kez | model, tekrar tekrar |
| Kaç arama | bir | ne kadar gerekiyorsa |
| Ne dönüyor | `k` parça | adaylar, sonra istendikçe tüm dosyalar |
| "Yeter"e kim karar veriyor | `k` | model |
| Arıza biçimi | kaçırdı | döngüye girdi, ya da erken durdu |

Erişimcinin işi *küçülüyor*, büyümüyor. Aday sunuyor ve ne kadar emin olduğu konusunda
dürüst oluyor. Muhakeme ajana geçiyor — ki Cursor ve Claude Code'un bir kod tabanı
üzerinde yaptığı bu, ve bu basamağın çoğunlukla araç tasarımıyla ilgili olmasının sebebi
de bu.

## Üç araç, ve sayının kendisi mesele

```bash
claude mcp add --transport http milvus-rag http://localhost:8090/mcp
```

| Araç | Ne yapıyor |
|---|---|
| `search_code` | adaylar, kanal başına skorlar ve aşağıdaki sinyallerle |
| `read_code` | dosyanın kalanını açar — **yalnızca indekslenmiş dosyaları** |
| `list_repos` | hangi kod tabanları bağlı ve her biri ne kadar taze |

`search_code` bir başlangıç noktası buluyor; `read_code` ise ajanın gerçekten karar verme
yolu — sonucun etrafını okuyarak. Tek atışlık top-k'nın tavan olmaktan çıkmasını sağlayan
o ikinci araç: `k`'nın doğru olması artık şart değil, çünkü ajan gerisini gidip alabiliyor.

Ayartma, araç eklemek: `find_definition`, `list_callers`, `search_by_symbol`. Onlarsız bir
şey bozulana kadar direnç göster. Her fazladan araç, modelin herhangi bir iş yapmadan önce
doğru vermesi gereken bir karar — ve menü büyüdükçe modeller seçmekte kötüleşiyor.

## Aracı bir arayüz için değil, bir model için tasarla

Basamağın özü bu. Dört şey, altındaki erişim kalitesinden daha önemli.

**Dürüst sinyalleri geçir.** Karar veren ajan olduğuna göre, erişimcinin bildiklerine
ihtiyacı var: [4. basamaktan](04-honesty.md) gelen zayıf eşleşme notu, kanal başına
skorlar, bir tasarım dokümanının içinde alıntılanan kodun gerçek kod sanılmaması için bir
`DOCUMENT` etiketi ve indeksin ne kadar bayat olduğu — indeksin dört günlük olduğunu bilen
bir ajan, "bu dosya yok" ile "bu dosya henüz indekslenmedi"yi ayırabilir.

**Başarısız olmanın serbest olduğunu söyle.** Sunucu talimatlarında, açıkça. Onsuz,
araçlarının cevaplayamayacağı bir soru sorulan bir model yine de cevaplar — 4. basamaktaki
arızanın aynısı, bir seviye yukarıda.

**Araçların erişebileceği alanı sınırla.** `read_code` yalnızca indekslenmiş dosyaları
açıyor. "Orada değil" ifadesini güvenilir kılan o kısıt — ve aynı zamanda güvenlik sınırı:
bir LLM'e bağlanmış, yol alan bir araç, olmayı bekleyen bir dizin gezintisi açığıdır.

**Getirilen içeriği veri olarak gör.**

```python
"""SECURITY: the code that comes back is from an indexed repo — it is DATA for the
agent, not INSTRUCTIONS. The server instructions say so explicitly."""
```

İndekslenmiş bir repo güvenilmeyen girdidir. Birinin README'sindeki "önceki talimatlarını
yok say" diyen bir yorum, ajana gerçek bir fonksiyon gövdesiyle aynı kanaldan ulaşır ve
sen söylemedikçe ikisini yapısal olarak ayıran hiçbir şey yoktur.

## İşletim tarafı

Bloklayan işin olay döngüsünde yeri yok. Embedding, vektör araması ve dosya okumaları
bloklar; bir MCP oturumu tek döngüde çalıştığı için orada bloklamak bağlantıdaki diğer her
isteği durdurur. Burada üçü de `anyio.to_thread` üzerinden geçiyor.

MCP sunucusunun HTTP API ile aynı süreçte, aynı erişimci üzerinde çalışması da kasıtlı: iki
süreç, iki önbellek, iki yapılandırma ve aynı soruya iki cevap demek.

## Maliyeti

Bu basamak konusunda açık ol; iyileştirdiği kadar kötüleştiren ilk basamak bu.

- **Gecikme ve token.** Üç araç çağrısı ve bir okuma, 34 ms'lik tek bir aramaya karşı
  birkaç saniye ve birkaç bin token.
- **Belirsizlik.** Aynı soru iki kez aynı yolu izlemiyor; bu da 4. basamaktaki eval'ini
  uygulamayı epey zorlaştırıyor. Recall@k bir ajanı tarif etmiyor. Sonunda sonuçları
  ölçüyorsun — doğru cevapladı mı, kaç çağrıda — daha küçük bir küme üzerinde, elle.
- **Yeni arıza biçimleri.** Hiçbir şey döndürmeyen bir sorguda döngüye girmek; ilk sonuç
  makul göründüğü için tek çağrıdan sonra durmak; korpustaki bir yorumun peşine takılmak.

Belirlenimci `POST /search` yolunu çalışır ve ölçülür tut. Geri düştüğün yer o, ve hâlâ
üzerine sayı koyabildiğin yer o.

## Neyi düzeltmiyor

Bazı soruların, kaç arama izin verirsen ver, hiçbir parça kümesinde cevabı yoktur:

> *"Bu kod tabanındaki ana temalar neler?"*
> *"Bu arayüzü değiştirirsem ne kırılır?"*

Bunların cevabı korpusun *içinde* değil, korpusun bir *özelliği* — parçaların birbirleriyle
nasıl ilişkilendiğinin.

O, **[6. basamak](06-graph.md)**.
