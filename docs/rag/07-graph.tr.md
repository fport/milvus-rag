# 7. basamak — GraphRAG

!!! done "Burada kurulmadı"

    Bu proje 6. basamakta duruyor. Bu sayfa 7. basamağın ne olduğunu, ne zaman değdiğini
    ve kurulacak olsaydı önce neyin kurulacağını anlatıyor — gerisiyle aynı ölçüde, ki bu
    da aşağıdaki sayıların başkalarına ait olduğu ve öyle etiketlendiği anlamına geliyor.

## Top-k'nın cevaplayamadığı soru

Buraya kadar her şey *pasaj* getiriyor. Bazı soruların hiçbir pasajda cevabı yok:

> *"Bu korpustaki ana temalar neler?"*
> *"Bu değişiklik hangi ekipleri etkiliyor?"*
> *"Bu arayüzü değiştirirsem ne kırılır?"*

Sıralama kötü olduğu için değil — cevap hiçbir yerde yazılı olmadığı için. Parçaların
birbirleriyle nasıl ilişkilendiğinin bir özelliği ve onu kurmak için hepsini okuman
gerekirdi. Microsoft'un GraphRAG makalesi bunu **yerel** sorular (birkaç pasajla
cevaplanır) ile **küresel** sorular (bir bütün olarak korpusla cevaplanır) farkı diye
adlandırıyor.

## Ne yapıyor

Metin sürümü, kabaca:

1. **Çıkar.** Bir LLM her parçayı okuyup varlıkları ve ilişkileri çekiyor —
   `(PaymentService) --[calls]--> (LedgerAPI)`.
2. **Grafiği kur**, varlıkları parçalar arasında birleştirerek. Bilgiyi yaratan adım bu:
   farklı dosyalardaki iki olgu artık tek bir kenar.
3. **Toplulukları tespit et** (Leiden ya da benzeri) ve LLM'e her kümenin özetini, sonra
   kümelerin kümelerinin özetini yazdır.
4. **Sorgula**, iki yoldan biriyle: **yerel** — varlığı bul, komşuluğunu dolaş, oradan
   cevapla; **küresel** — topluluk özetlerinden cevapla ve birleştir.

Bütün numara topluluk özetlerinde. Küresel bir soru, "korpusun bu bölümünde ne var"ın
önceden hesaplanmış bir hiyerarşisinden cevaplanıyor, asla korpusu getirerek değil.

## Kodda grafik zaten var

Hesabı değiştiren kısım burası ve GraphRAG yazılarının çoğu bunu kaçırıyor, çünkü düzyazı
hakkında yazılmışlar.

Bir kod tabanının **hiçbir LLM gerektirmeyen** gerçek ve kesin bir grafiği var: import'lar,
çağrılar, tip referansları, tanımlar ve kullanımları. Bu projede tree-sitter zaten her
dosyayı ayrıştırıyor — parçaları üreten aynı geçiş, kenarları da belirlenimci biçimde ve
bedavaya üretebilir.

Bu önemli, çünkü 7. basamağın asıl maliyetini ortadan kaldırıyor. Metin GraphRAG'inde
çıkarım, korpusun tamamı üzerinde bir LLM geçişi; gerçek paraya mal oluyor, *olasılıksal*
(aynı doküman iki kez çıkarıldığında farklı varlıklar veriyor) ve korpus değiştiğinde
yeniden koşulması gerekiyor. Bir çağrı grafiği bunların hiçbiri değil: kesin, ucuz ve
burada indekslemeyi zaten süren aynı dosya bazlı farkla artımlı güncelleniyor.

Yani kod için mantıklı sıra "koda uygulanmış metin GraphRAG'i" değil. Şu:

**Önce bedava yapısal grafik.** Sembol → tanım, tanım → referanslar, dosya → import'lar.
Sonra bunu erişimi *genişletmek* için kullan, yerine geçirmek için değil: 3. basamağın
sonuçlarını al, bir adım ötedeki komşularını ekle, ajana ikisini birden ver. Bu "buna kim
çağırıyor" ve "ne kırılır" sorularını cevaplıyor — ki pratik talebin çoğu bu — ve arama
başına fazladan bir sorguya mal oluyor.

**Ancak ondan sonra, ve yalnızca gerekiyorsa, LLM katmanı** — gerçekten küresel sorular
("ana alt sistemler neler") için topluluklar ve özetler. Maliyet ve bayatlık orada yaşıyor
ve yapısal grafik bir kez varsa çok daha küçük bir adım.

## Maliyeti

| Maliyet | Metin GraphRAG | Kod, yapısal grafik |
|---|---|---|
| Kurulum | her parça üzerinde bir LLM geçişi | bedava, zaten yaptığın ayrıştırmada |
| Değişimde yeniden kurulum | değişen bölgeyi yeniden çıkar, yeniden kümele | artımlı, kesin |
| Belirlenimlilik | olasılıksal çıkarım | kesin |
| Sorgu gecikmesi | topluluk özetleri, birden çok LLM çağrısı | fazladan bir arama |
| Değerlendirmek | zor — küresel cevapların tek bir doğru pasajı yok | sıradan Recall@k hâlâ geçerli |

Bunlardan ikisi vurgulanmayı hak ediyor.

**Maliyet.** Yayınlanmış GraphRAG indeksleme koşuları yaygın olarak korpusun milyon token'ı
başına onlarca dolar ve saatlerce duvar saati olarak anılıyor. *Burada* ölçülmüş bir şeyle
kalibrasyon için: LLM parça açıklamaları — çok daha basit, parça başına bir LLM geçişi —
yerel bir 9B'de 2.8k parçalık bir repo için yaklaşık dört saat sürdü. Grafik çıkarımı o
biçimde bir maliyet, üstüne bir de kümeleme geçişi.

**Değerlendirme.** Bu konuda yavaş olmanın asıl sebebi bu. Küresel bir cevabın altın pasajı
yok, dolayısıyla Recall@k geçerli değil ve dürüst yöntemler kapsamlılık ile çeşitlilik
üzerinde LLM-hakem, ya da insan puanlaması. İkisi de [5. basamaktaki](05-honesty.md)
araçlardan daha zayıf. 7. basamağı kurup işe yaradığını gösterememek tamamen mümkün — ki bu
projenin tek kuralına göre, yayınlamaman gerektiği anlamına gelir.

## Ne zaman çıkmalı

Şunlarda çık:

- başarısız olduğun sorular **küresel** — temalar, etki, "neye ne bağlı"
- korpusunda gerçek belgeler arası yapı var (bir kod tabanı, bir dava dosyası, gerçekten
  birbirine bağlı bir wiki), birbirinden bağımsız makale yığını değil
- korpus, grafiği yeniden kurmanın haftalık bir olay olmayacağı kadar durağan
- ve — kod için — bedava yapısal grafiği zaten aldın ve yetmedi

Gerçekten gördüğün arızalar "fonksiyonu bulamadı" ya da "reddetmesi gereken bir soruyu
cevapladı" ise çıkma. Bunlar 3. ve 5. basamak, çok daha ucuzlar ve 7. basamak onları
düzeltmiyor.

!!! measured "Bu basamağın dürüst durumu"

    Bu sayfadaki hiçbir şey bu korpusta ölçülmedi. Altındaki basamaklar ölçüldü ve ikisi
    beklentiye ters çıktı.

    Bu, 6. basamağa karşı değil merdivenin lehine bir argüman: en üst basamağın maliyetini
    bilmenin sebebi, daha ucuz olanların gerçekten tükenip tükenmediğini söyleyebilmek.
    Çoğu sistemde tükenmemiştir.
