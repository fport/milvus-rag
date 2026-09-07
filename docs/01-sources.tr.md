# 1. Kaynaklar ve senkronizasyon

Bir repoyu içeri almak ve güncel tutmak, kod RAG'inin önemsiz görünen ama olmayan
kısmıdır. Bayat bir indeksin erişim kalitesi, embedding'ler ne kadar iyi olursa olsun
sıfırdır.

!!! done "Bu aşamanın sahibi olduğu yerler"

    `sources/` (Azure DevOps, GitHub, git, dosya keşfi), `webhooks.py`, `jobs.py`,
    `db.py` içindeki SQLite tabloları ve `index/pipeline.py`'nin başındaki manifest
    farkı.

## Üç kaynak, tek yol

Azure DevOps ve GitHub'a proje ve repo listelemek için REST üzerinden gidiliyor;
klonlamanın kendisi düz git. Yerel bir dizin, klonlama adımı atlanarak aynı kod
yolundan geçiyor — servisi denemeyi tek satıra indiren şey de bu.

```bash
uv run rag azure projects
uv run rag azure repos Platform
uv run rag add-azure Platform backend-api --branch develop

uv run rag github repos sindresorhus        # açık repolar için token gerekmez
uv run rag add-github sindresorhus/p-limit

uv run rag add-local ~/code/my-api --name my-api   # git reposu bile olması gerekmiyor
```

### PAT hiçbir yere yazılmaz

Uzak URL'in içindeki bir kişisel erişim jetonu `.git/config`'e, `git remote -v`
çıktısına ve bir sonraki hata mesajının metnine düşer. Bu yüzden oraya konmuyor:

```python
# sources/git.py
"""The credential is never embedded in the URL and never written to the remote config:
it is passed to every command as `-c http.extraheader=AUTHORIZATION: Basic ...`."""

_SECRET = re.compile(r"(Basic|Bearer)\s+[A-Za-z0-9+/=_\-]+", re.IGNORECASE)
```

Hata çıktısı loglanmadan ya da döndürülmeden önce bu desenden geçiriliyor, çünkü
başarısız bir `git fetch` denediği komutu ekrana basar.

## Tazelik: arkasında yoklayıcı olan bir webhook

İki sağlayıcı da aynı `PushEvent`'e indirgeniyor ve her biri farklı doğrulanıyor,
çünkü her biri farklı bir şey sunuyor:

| Sağlayıcı | Doğrulama |
|---|---|
| Azure DevOps | paylaşılan sır — Basic auth parolası, `X-RAG-Webhook-Secret` ya da `?secret=` |
| GitHub | gövdenin HMAC-SHA256'sı (`X-Hub-Signature-256`), aynı sırra karşı |

Tek başına webhook bir tazelik garantisi değil: kaçan bir teslimat kalıcı bir boşluktur
ve sistemde bunu fark edecek hiçbir şey yoktur. Bu yüzden yoklayıcı her
`RAG_POLL_INTERVAL_SECONDS` aralığında uzak daldaki başı en son indekslenen commit ile
karşılaştırıyor. Tek başına cron bayat bir pencere demek olurdu; tek başına webhook
sessiz boşluklar. İkisi birlikte, aralık başına bir fazladan HTTP çağrısına mal oluyor.

Yalnızca izlenen dala gelen bir push iş açıyor; tekrarlanan bir `(repo, commit)` —
Azure yeniden dener — hiçbir şey açmıyor.

## Tek işçi, repo başına tek bekleyen iş

```python
# jobs.py
"""Why a single worker: indexing is CPU-bound (embedding) and writes to Milvus;
indexing two repos at once slows both down and speeds up neither."""
```

Repo başına tek bekleyen iş garantisi var — BullMQ'nun job-id tekilleştirmesiyle aynı
fikir: bir dakikada aynı repoya beş push gelirse bir iş çalışır, bir iş bekler. Diğer
üçü kuyruğa girmez, çünkü ilki bitene kadar hepsi aynı commit'i indeksliyor olacaktır.

İş kayıtları SQLite'ta duruyor. Yeniden başlatmadan sonra `running` kalan işler `failed`
oluyor, `queued` olanlar yeniden kuyruğa giriyor — indeksleme ortasında bir çökme,
sessizce kaybolan bir iş değil, görünür bir başarısız iştir.

## İçerik özetiyle değişiklik tespiti

Bütün bir hata kategorisini ortadan kaldıran karar bu.

```python
# index/pipeline.py
"""Change detection is based on the content hash, not on a commit diff: there are no
git edge cases from renames, mode changes or submodules, and local (non-git)
directories take the same path."""
```

İndekslenebilir her dosyanın sha256'sı SQLite `files` tablosunda tutuluyor. Bir
senkronizasyon çalışma kopyasını hash'liyor ve karşılaştırıyor:

| Sonuç | Ne oluyor |
|---|---|
| eklenmiş | parçala, embed'le, ekle |
| değişmiş | o yoldaki parçaları sil, sonra yenilerini ekle |
| silinmiş | yalnızca sil |
| değişmemiş | **bir daha asla embed'lenmez** |

`git diff` daha ucuz olurdu ve beraberinde yeniden adlandırmaları, mod değişikliklerini,
force-push'ları ve submodule güncellemelerini getirirdi. Her dosyayı hash'lemek beş bin
dosya için bir saniyenin altında — embedding'in yanında görünmez.

### Kesintiden sağ çıkmak

Buradaki sıralama kasıtlı ve söylenmeye değer, çünkü akla ilk gelen sıralama bir delik
bırakıyor:

```
manifest satırını kaldır  →  eski parçaları sil  →  yenilerini yaz  →  satırı geri koy
```

Süreç ortada bir yerde ölürse dosya manifestte *yok* olur, bu yüzden bir sonraki
senkronizasyon onu yeni sayar ve yeniden indeksler. Satır önce güncellenseydi,
"güncelle" ile "yaz" arasındaki bir çökme, manifestin güncel dediği ama indekste
bulunmayan bir dosya bırakırdı — kalıcı ve sessiz bir boşluk.

## Ne indeksleniyor

Bir repodaki her şey embed'lenmeye değmez. `sources/files.py` dizine, uzantıya ve
boyuta göre filtreliyor:

- `IGNORED_DIRS` — `node_modules`, `.git`, `dist`, derleme çıktısı, sanal ortamlar
- `IGNORED_SUFFIXES` — ikili dosyalar, görseller, arşivler, kilit dosyaları
- `MAX_JSON_BYTES = 64_000` — büyük bir JSON dosyası kod değil, veridir

Kalan her dosya bir `lang` ve bir `category` (code, document, config) alıyor; erişimci
bunu sonradan `DOCUMENT` rozeti olarak gösteriyor — böylece bir ajan, bir tasarım
dokümanının içindeki kod parçasını gerçek koddan ayırabiliyor.

## Yazımdan sonra

Bir iş bittiğinde iki şey oluyor:

**Arama önbelleği temizleniyor.** Aksi hâlde önbellekten cevaplanan bir sorgu önceki
commit'in satır numaralarını göstermeye devam eder — erişim hatası gibi görünen ama
olmayan bir arıza.

**`index_version` kontrol ediliyor.** Bir `RAG_*` parçalama ya da embedding ayarı
değiştiyse versiyon da onunla değişiyor ve bir sonraki senkronizasyon artımlı yerine
tam yeniden indeksleme yapıyor. Farklı ayarlarla üretilmiş bir parça, mevcut ayarlarla
üretilmiş olanla kıyaslanabilir değildir; ikisini karıştırmak her sonucu sessizce
bozar.

```bash
uv run rag sync my-api              # artımlı
uv run rag sync my-api --force      # tam yeniden indeksleme
uv run rag poll                     # uzak dal başlarını bir kez kontrol et
```
