<p align="center">
  <img src="docs/screenshot.png" alt="Milvus RAG arayüzü — Türkçe soruyla kod araması, kanal skorları ve repo durumu" width="920">
</p>

<h1 align="center">Milvus RAG</h1>

<p align="center"><em>Kod tabanını anlayan, push'ta kendini tazeleyen arama servisi.</em></p>

<p align="center">
  <a href="#kurulum">Kurulum</a> ·
  <a href="#kullanım">Kullanım</a> ·
  <a href="#repo-değişince-ne-olur">Tazeleme akışı</a> ·
  <a href="#retrieval-zinciri-ve-bayraklar">Retrieval</a> ·
  <a href="#ölçüm-defteri">Ölçüm defteri</a> ·
  <a href="DEPLOYMENT.md">Sunucuya kurulum</a>
</p>

---

Azure DevOps'tan ya da GitHub'dan bir repo seçersin; servis
onu klonlar, tree-sitter ile kod birimlerine böler, BGE-M3 (dense) + BM25 (sparse)
ile Milvus'a yazar; sorguda sembol biçimli sorgular BM25'e, düz cümleler dense'e gider
(hybrid RRF ve cross-encoder rerank bayrakla açılır — ikisi de ölçüldü, tabloya bak).
Repoya push geldiğinde yalnızca değişen dosyalar yeniden indexlenir.

```
Azure / GitHub ──git clone/fetch──▶ çalışma kopyası ──sha256 manifest──▶ değişen dosyalar
   │                                                                        │
   │ webhook (git.push / push + HMAC)  /  poller                          tree-sitter chunk
   ▼                                                                        │
/webhooks/azure/push ──▶ iş kuyruğu ──▶ sil + yeniden yaz ──▶ Milvus (dense + BM25, repo_id partition)
                                                                            │
POST /search  ──▶ sembol mü? ──▶ BM25 ─────────────────────────┐            │
              └▶ düz cümle ──▶ dense ─(bayrak: +BM25→RRF, rerank)─┴─▶ 8 hit (kanal skorlarıyla)
POST /ask     ──▶ /search + LLM (atıflı cevap)
```

## Kurulum

```bash
git clone <repo> && cd <repo>
uv sync                                              # Python 3.12 + bağımlılıklar (uv indirir)
docker compose -f infra/docker-compose.yml up -d     # Milvus 2.6 + etcd + MinIO
curl -f http://localhost:9091/healthz                # "OK" (ilk açılış ~60-90 sn)
cp .env.example .env                                 # aşağıdaki değerleri doldur
```

`.env`'de en az:

| Değişken | Ne |
|---|---|
| `AZURE_DEVOPS_ORG_URL` | `https://dev.azure.com/<org>` |
| `AZURE_DEVOPS_PAT` | Personal Access Token, kapsam **Code → Read** |
| `GITHUB_TOKEN` | isteğe bağlı — public repolar tokensız çalışır; private için fine-grained PAT (Contents: Read) |
| `RAG_WEBHOOK_SECRET` | Azure Service Hook'un göndereceği paylaşılan sır |
| `ANTHROPIC_API_KEY` | yalnızca `/ask` ve enrichment için (retrieval LLM'siz çalışır) |

İlk çalıştırmada `BAAI/bge-m3` (~2.2 GB) ve `BAAI/bge-reranker-v2-m3` (~2.2 GB)
Hugging Face'ten iner; sonrası `~/.cache/huggingface`'ten gelir. Apple M-serisinde
otomatik `mps` kullanılır.

## Kullanım

```bash
uv run rag serve   # http://localhost:8090 → web arayüzü · /docs → OpenAPI

# Azure'a göz at, repo seç
uv run rag azure projects
uv run rag azure repos Platform
uv run rag add-azure Platform backend-api         # kaydeder + klonlar + indexler
uv run rag add-azure Platform backend-api --branch develop

# GitHub'dan repo seç (public için token gerekmez)
uv run rag github repos sindresorhus
uv run rag add-github sindresorhus/p-limit
uv run rag add-github acme/backend --branch develop

# Yerel bir dizinle dene (git olması şart değil)
uv run rag add-local ~/code/my-api --name my-api

# Ara / sor / ölç
uv run rag search "webhook eventleri nasıl kuyruğa alınıyor" -r my-api
uv run rag search withSession --mode bm25
uv run rag ask "optimistic lock nasıl çalışıyor?" -r my-api
uv run rag eval evals/golden.example.jsonl -r my-api --tag v1   # şablonu kopyalayıp doldur

# Tazele
uv run rag sync my-api                            # artımlı: yalnızca değişen dosyalar
uv run rag sync my-api --force                    # tam yeniden index
uv run rag poll                                   # Azure head'lerini bir kez kontrol et
```

Aynı işlemler HTTP'den (`/` altında basit bir web arayüzü de var: repo ekle,
index ilerlemesi, kanal skorlarıyla arama, LLM'e soru):

| Uç | İş |
|---|---|
| `GET /health` | Milvus, model ve Azure durumu |
| `GET /azure/projects`, `GET /azure/projects/{p}/repos` | Azure'a göz at (`registered_as` alanı kayıtlıysa dolu) |
| `GET /github/{owner}/repos` | GitHub org/kullanıcı repoları (aynı `registered_as` alanıyla) |
| `POST /repos` | kaydet + indexle — `{provider:"azure", project, repo}` · `{provider:"github", repo:"owner/repo"}` · `{provider:"local", path}` |
| `GET /repos`, `GET /repos/{id}`, `DELETE /repos/{id}` | durum (aktif iş dahil), silme |
| `POST /repos/{id}/sync` `{force?}` | iş kuyruğa alınır → 202 |
| `GET /repos/{id}/files`, `GET /repos/{id}/file?path=&start=&end=` | manifest; dosyadan satır aralığı (agent `read_file`) |
| `GET /jobs`, `GET /jobs/{id}` | iş geçmişi ve ilerleme (`stats.progress`) |
| `POST /search` | `{q, repo_ids?, k?, mode?, rerank?, candidates?, path_prefix?, lang?, category?}` |
| `POST /ask` | aynı gövde → `{answer, sources[]}` |
| `POST /webhooks/azure/push` | Azure Service Hook hedefi |
| `POST /webhooks/github/push` | GitHub webhook hedefi (HMAC-SHA256 imza doğrulanır) |

Her `/search` sonucu kanal skorlarını taşır — hangi kanalın neyi bulduğu görünür:

```json
{"path": "src/features/jira-webhook/jira-webhook.queue.ts", "symbol": "enqueueWebhookEvent",
 "start_line": 41, "end_line": 62,
 "scores": {"dense": 0.71, "bm25": 14.2, "rrf": 0.0325, "rerank": 0.93}}
```

## Repo değişince ne olur

1. **Tetik.**
   - *Azure:* Project Settings → Service Hooks → *Web Hooks* → olay **Code pushed**,
     URL `https://<host>:8090/webhooks/azure/push`, "Basic authentication password" ya da
     `X-RAG-Webhook-Secret` header'ı = `RAG_WEBHOOK_SECRET`.
   - *GitHub:* repo → Settings → Webhooks → Add webhook: URL
     `https://<host>:8090/webhooks/github/push`, content type `application/json`,
     **Secret** = `RAG_WEBHOOK_SECRET` (HMAC-SHA256 imzası doğrulanır), olay: *Just the push event*.
   - Webhook yoksa/kaçarsa poller `RAG_POLL_INTERVAL_SECONDS` aralığıyla Azure/GitHub
     branch head'ini karşılaştırır.
2. **Kuyruk.** Repo başına tek bekleyen iş: beş push gelirse bir iş çalışır, bir iş bekler.
   Aynı commit ikinci kez gelirse (Azure yeniden dener) iş açılmaz.
3. **Fark.** `git fetch` + `reset --hard origin/<branch>`, sonra her indexlenebilir dosyanın
   sha256'sı SQLite manifest ile karşılaştırılır: eklenen / değişen / silinen / aynı.
4. **Yazma.** Değişen dosya için önce o yolun chunk'ları silinir, sonra yenileri eklenir;
   silinen dosya yalnızca silinir; aynı kalan dosya embed edilmez. Manifest satırı silmeden
   önce kaldırılır, yazdıktan sonra geri konur — yarıda kesilen iş eksik bırakmaz.
5. **Cache.** İş bitince arama cache'i boşalır; eski satır numaralarını gösteren cevap kalmaz.

`RAG_*` chunk/embedding ayarı değişirse `index_version` değişir ve bir sonraki sync tam
yeniden index yapar.

## Retrieval zinciri ve bayraklar

| Bayrak | Varsayılan | Ne yapar |
|---|---|---|
| `RAG_SEARCH_MODE` | `auto` | sembol biçimli sorgu (`handleAuthCallback`, `QUEUE_NAMES`, `a.b.c`) → BM25; düz cümle → `RAG_PROSE_MODE` |
| `RAG_PROSE_MODE` | `dense` | `dense` ya da `hybrid` (dense + BM25 → RRF, `RAG_RRF_K=60`) — ölçüldü, aşağıya bak |
| `RAG_RERANK_ENABLED` | `false` | `RAG_CANDIDATES=40` aday → `bge-reranker-v2-m3` → `RAG_TOP_K=8` — ölçüldü, zarar etti |
| `RAG_CHUNK_MAX_BYTES` / `MIN` | 2000 / 200 | chunk sınırları (≈500 token tavan) |
| `RAG_ENRICH_ENABLED` | `false` | her chunk için LLM'den Türkçe açıklama, `indexed_text`'e girer (cache'li) |

Her bayrağı `POST /search` gövdesinde ve `rag eval` parametrelerinde istek başına ezebilirsin;
ablation için tasarlandı.

### Ölçüm defteri

`rag eval` her çalışmada `evals/results/<tarih>_<etiket>.json` yazar. Golden set
şablonu `evals/golden.example.jsonl`; kendi repon için kopyalayıp doldur (repo'ya
özel setler gitignore'da — iç dosya yollarını yayınlama). Aşağıdaki rakamlar örnek
bir korpus üzerinde (318 dosyalık TypeScript API monorepo'su, 42 soru: 19 EN düz,
19 TR düz, 4 sembol; etiketler repo okunarak yazıldı), k=8:

İlk ölçüm (2026-08-31, BGE-M3, chunk 200–2000B):

| Etiket | Ayar | Recall@8 | MRR | TR-prose R@8 | p50 |
|---|---|---|---|---|---|
| bm25 | yalnız BM25 | 0.405 | 0.240 | 0.263 | 2 ms |
| hybrid | dense+BM25 → RRF | 0.786 | 0.604 | 0.684 | 40 ms |
| dense | yalnız dense | 0.786 | 0.678 | 0.684 | 32 ms |
| **auto+dense** ✓ | sembol→bm25, düz→dense | **0.786** | **0.690** | 0.684 | 34 ms |
| hybrid k=40 | aday havuzu | 0.952 | — | 0.895 | 39 ms |
| hybrid+rerank | 40→8, bge-reranker-v2-m3 | 0.762 | 0.508 | 0.579 | 4389 ms |
| auto+rerank | aynı | 0.762 | 0.514 | 0.579 | 2050 ms |

Okuma: sembollerde BM25 1.0/1.0, dense 1.0/0.875 — yönlendirme MRR'ı bedavaya taşıyor.
BGE-M3 çok dilli olduğu için Türkçe sorular çalışıyor (eski MiniLM deneyinde 0.04'tü).
Recall@40 = 0.95: reranker'ın kapatabileceği +0.17'lik alan VAR ama bge-reranker-v2-m3
onu kapatmak yerine sıralamayı bozdu ve saniyeler yedi → kapalı. Daha iyi bir reranker
denenecekse tabloya yeni satır olarak girer.

```bash
G=evals/golden.example.jsonl   # kendi setinle değiştir
uv run rag eval $G -r my-api --tag dense  --mode dense  --no-rerank
uv run rag eval $G -r my-api --tag bm25   --mode bm25   --no-rerank
uv run rag eval $G -r my-api --tag hybrid --mode hybrid --no-rerank
uv run rag eval $G -r my-api --tag rerank --mode auto   --rerank
```

## Tasarım notları

- **Neden içerik hash'i, commit diff'i değil?** Yeniden adlandırma, mod değişikliği,
  submodule, force-push gibi kenar durumları yok; yerel (git olmayan) dizinler de aynı
  yoldan geçer. Maliyet: her sync tüm dosyaları hash'ler — 5k dosyada bir saniyenin altı,
  embedding'in yanında görünmez.
- **Neden tek Milvus collection?** `repo_id` partition key; `repo_id in [...]` filtresi
  yalnızca ilgili partition'lara iner. Repo başına collection açmak çapraz-repo aramayı
  zorlaştırır ve collection sayısını sınırlar.
- **Neden sembol sorguları BM25'e gidiyor?** Önceki deney (production-ready-rag-system,
  ADR-0002) aynı korpusta ölçtü: sembol aramasında BM25 0.80, dense 0.60, ikisinin RRF'i
  0.60 — bulamayan kanal da tam güçle terfi ediyor. Bir regex bunu bedavaya çözer.
- **Neden reranker kapalı?** Ölçüldü: bge-reranker-v2-m3 bu korpusta hem recall hem MRR
  düşürdü (özellikle Türkçe'de) ve p50'yi 2-4 sn yaptı. Açıksa 40 aday alır — 8 adayı
  yeniden sıralamak recall'a dokunamaz; Recall@40 − Recall@8 farkı reranker'ın çalışma
  alanıdır. Fark bu korpusta var (0.95 − 0.79); onu kapatan bir model bulunursa tabloya
  satır olarak girer.
- **Neden enrichment kapalı?** Deneyde İngilizce kod üstünde Türkçe soru recall@5 = 0.04
  çıktı ve LLM açıklamaları çare oldu; ama o deney İngilizce-only MiniLM iledi. BGE-M3 çok
  dilli. Önce ölç, gerekiyorsa aç.
- **tree-sitter'da iki tuzak (ölçüldü).** py-tree-sitter 0.26 + language-pack 1.15 ile
  `Node.start_point/end_point` okumak uzun süreçte segfault veriyor (aynı dosya tek başına
  geçiyor, 39. dosyada çöküyor); satır numaraları byte ofsetinden bisect ile hesaplanır.
  `tree-sitter-sql` drizzle migration dosyalarında doğrudan çöküyor; `.sql` kod sayılır ama
  paragraf/satır pencereleriyle chunk'lanır (`CODE_WITHOUT_GRAMMAR`).
- **PAT güvenliği.** Token URL'ye gömülmez; git'e `-c http.extraheader=` ile geçer,
  `.git/config`'e yazılmaz, hata mesajlarında redakte edilir. İndex'e girmeden önce
  `scrub` (API anahtarı, JWT, e-posta, `X_PASSWORD=...`) çalışır.

## Geliştirme

```bash
uv run pytest -q                              # birim testleri (Milvus/model gerekmez)
RAG_LIVE=1 uv run pytest -q tests/test_milvus_live.py   # gerçek Milvus'a karşı depo testi
uv run ruff check src tests && uv run ruff format --check src tests
```

Proje düzeni ve bağlayıcı kararlar için `CLAUDE.md`.

## Bir agent'a / uygulamaya bağlama

Servis düz HTTP konuşur; bir asistana bağlamak için iki araç yeter:

- `search_code(query, repo?)` → `POST /search` — kanal skorlarıyla ilk adaylar
- `read_code(repo, path, start, end)` → `GET /repos/{id}/file` — agent gerisini okuyarak karar verir

Sunucuya kurulum için `DEPLOYMENT.md`. Lisans: MIT.
