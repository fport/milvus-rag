# 7. basamak — GraphRAG

!!! done "Burada kurulmadı"

    Bu proje 6. basamakta duruyor. Bu sayfa 7. basamağın ne olduğunu, ne zaman değdiğini
    ve kurulacak olsaydı önce neyin kurulacağını anlatıyor — gerisiyle aynı ölçüde, ki bu
    da aşağıdaki sayıların başkalarına ait olduğu ve öyle etiketlendiği anlamına geliyor.

## Top-k'nın cevaplayamadığı soru

Buraya kadar her şey *pasaj* getiriyor. Bazı soruların hiçbir pasajda cevabı yok:

> *"Bu corpus'taki ana temalar neler?"*
> *"Bu değişiklik hangi ekipleri etkiliyor?"*
> *"Bu arayüzü değiştirirsem ne kırılır?"*

Sıralama kötü olduğu için değil — cevap hiçbir yerde yazılı olmadığı için. Chunkların
birbirleriyle nasıl ilişkilendiğinin bir özelliği ve onu kurmak için hepsini okuman
gerekirdi. Microsoft'un GraphRAG makalesi bunu **local** sorular (birkaç pasajla
cevaplanır) ile **global** sorular (bir bütün olarak corpus'la cevaplanır) farkı diye
adlandırıyor.

## Ne yapıyor

Metin sürümü, kabaca:

1. **Çıkar.** Bir LLM her chunk'ı okuyup varlıkları ve ilişkileri çekiyor —
   `(PaymentService) --[calls]--> (LedgerAPI)`.
2. **Grafiği kur**, varlıkları chunk'lar arasında birleştirerek. Bilgiyi yaratan adım bu:
   farklı dosyalardaki iki olgu artık tek bir kenar.
3. **Toplulukları tespit et** (Leiden ya da benzeri) ve LLM'e her kümenin özetini, sonra
   kümelerin kümelerinin özetini yazdır.
4. **Sorgula**, iki yoldan biriyle: **local** — varlığı bul, komşuluğunu dolaş, oradan
   cevapla; **global** — topluluk özetlerinden cevapla ve birleştir.

Bütün numara topluluk özetlerinde. Global bir soru, "corpus'un bu bölümünde ne var"ın
önceden hesaplanmış bir hiyerarşisinden cevaplanıyor, asla corpus'u getirerek değil.

## Kodda grafik zaten var

Hesabı değiştiren kısım burası ve GraphRAG yazılarının çoğu bunu kaçırıyor, çünkü düzyazı
hakkında yazılmışlar.

Bir kod tabanının **hiçbir LLM gerektirmeyen** gerçek ve kesin bir grafiği var: import'lar,
çağrılar, tip referansları, tanımlar ve kullanımları. Bu projede tree-sitter zaten her
dosyayı ayrıştırıyor — chunk'ları üreten aynı geçiş, kenarları da deterministik biçimde ve
bedavaya üretebilir.

Bu önemli, çünkü 7. basamağın asıl maliyetini ortadan kaldırıyor. Metin GraphRAG'inde
çıkarım, corpus'un tamamı üzerinde bir LLM geçişi; gerçek paraya mal oluyor, *olasılıksal*
(aynı doküman iki kez çıkarıldığında farklı varlıklar veriyor) ve corpus değiştiğinde
yeniden koşulması gerekiyor. Bir çağrı grafiği bunların hiçbiri değil: kesin, ucuz ve
burada indekslemeyi zaten süren aynı dosya bazlı farkla artımlı güncelleniyor.

Yani kod için mantıklı sıra "koda uygulanmış metin GraphRAG'i" değil. Şu:

**Önce bedava yapısal grafik.** Sembol → tanım, tanım → referanslar, dosya → import'lar.
Sonra bunu retrieval'ı *genişletmek* için kullan, yerine geçirmek için değil: 3. basamağın
sonuçlarını al, bir adım ötedeki komşularını ekle, ajana ikisini birden ver. Bu "buna kim
çağırıyor" ve "ne kırılır" sorularını cevaplıyor — ki pratik talebin çoğu bu — ve arama
başına fazladan bir sorguya mal oluyor.

**Ancak ondan sonra, ve yalnızca gerekiyorsa, LLM katmanı** — gerçekten global sorular
("ana alt sistemler neler") için topluluklar ve özetler. Maliyet ve bayatlık orada yaşıyor
ve yapısal grafik bir kez varsa çok daha küçük bir adım.

## Maliyeti

| Maliyet | Metin GraphRAG | Kod, yapısal grafik |
|---|---|---|
| Kurulum | her chunk üzerinde bir LLM geçişi | bedava, zaten yaptığın ayrıştırmada |
| Değişimde yeniden kurulum | değişen bölgeyi yeniden çıkar, yeniden kümele | artımlı, kesin |
| Determinizm | olasılıksal çıkarım | kesin |
| Sorgu gecikmesi | topluluk özetleri, birden çok LLM çağrısı | fazladan bir arama |
| Değerlendirmek | zor — global cevapların tek bir doğru pasajı yok | sıradan Recall@k hâlâ geçerli |

Bunlardan ikisi vurgulanmayı hak ediyor.

**Maliyet.** Yayınlanmış GraphRAG indeksleme koşuları yaygın olarak corpus'un milyon token'ı
başına onlarca dolar ve saatlerce duvar saati olarak anılıyor. *Burada* ölçülmüş bir şeyle
kalibrasyon için: LLM chunk açıklamaları — çok daha basit, chunk başına bir LLM geçişi —
yerel bir 9B'de 2.8k chunk'lık bir repo için yaklaşık dört saat sürdü. Grafik çıkarımı o
biçimde bir maliyet, üstüne bir de kümeleme geçişi.

**Değerlendirme.** Bu konuda yavaş olmanın asıl sebebi bu. Global bir cevabın gold pasajı
yok, dolayısıyla Recall@k geçerli değil ve dürüst yöntemler kapsamlılık ile çeşitlilik
üzerinde LLM-hakem, ya da insan puanlaması. İkisi de [5. basamaktaki](05-honesty.md)
araçlardan daha zayıf. 7. basamağı kurup işe yaradığını gösterememek tamamen mümkün — ki bu
projenin tek kuralına göre, yayınlamaman gerektiği anlamına gelir.

## Ne zaman çıkmalı

Şunlarda çık:

- başarısız olduğun sorular **global** — temalar, etki, "neye ne bağlı"
- corpus'unda gerçek belgeler arası yapı var (bir kod tabanı, bir dava dosyası, gerçekten
  birbirine bağlı bir wiki), birbirinden bağımsız makale yığını değil
- corpus, grafiği yeniden kurmanın haftalık bir olay olmayacağı kadar durağan
- ve — kod için — bedava yapısal grafiği zaten aldın ve yetmedi

Gerçekten gördüğün arızalar "fonksiyonu bulamadı" ya da "reddetmesi gereken bir soruyu
cevapladı" ise çıkma. Bunlar 3. ve 5. basamak, çok daha ucuzlar ve 7. basamak onları
düzeltmiyor.

!!! measured "Bu basamağın dürüst durumu"

    Bu sayfadaki hiçbir şey bu corpus'ta ölçülmedi. Altındaki basamaklar ölçüldü ve ikisi
    beklentiye ters çıktı.

    Bu, 6. basamağa karşı değil merdivenin lehine bir argüman: en üst basamağın maliyetini
    bilmenin sebebi, daha ucuz olanların gerçekten tükenip tükenmediğini söyleyebilmek.
    Çoğu sistemde tükenmemiştir.
