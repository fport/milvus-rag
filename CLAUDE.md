# CLAUDE.md — milvus-rag

Kod tabanı RAG servisi. Azure DevOps'tan, GitHub'dan (ya da yerel dizinden) bir
repo seçilir, tree-sitter ile kod birimlerine bölünür, BGE-M3 + BM25 ile
Milvus'a yazılır, cross-encoder ile yeniden sıralanır; repo değişince yalnızca
değişen dosyalar yeniden indexlenir.

Bağımsız bir Python 3.12 + uv paketi; dış dünyayla yalnızca HTTP konuşur.
İndexlediği repolar salt-okunurdur — klonlara asla yazılmaz.

## Komutlar

```bash
uv sync                                              # ortam (Python 3.12'yi uv indirir)
docker compose -f infra/docker-compose.yml up -d     # Milvus + etcd + MinIO (:19530)
cp .env.example .env                                 # AZURE_DEVOPS_*, RAG_WEBHOOK_SECRET doldur

uv run rag serve                                     # FastAPI :8090  (docs: /docs)
uv run rag azure projects | uv run rag azure repos <proje>
uv run rag add-azure <proje> <repo> [--branch main]  # kaydet + indexle
uv run rag add-github <owner/repo> [--branch main]   # public için token gerekmez
uv run rag add-local ~/code/my-api --name my-api     # yerel dizin (test için)
uv run rag sync <repo-id> [--force]                  # artımlı / tam yeniden index
uv run rag search "..." [-r <repo-id>] [--mode bm25] [--no-rerank] [--json]
uv run rag ask "..."                                 # atıflı cevap (LLM gerekir)
uv run rag eval evals/golden.example.jsonl -r my-api --tag <etiket>
uv run pytest -q && uv run ruff check src tests      # testler + lint
```

## Mimari

```
sources/   azure.py + github.py (REST: repo/ref) · git.py (clone/fetch, PAT header ile) · files.py (dosya filtreleri, sha256)
index/     chunk.py (tree-sitter AST chunker + markdown + fallback) · scrub.py (sır/PII)
           embed.py (BGE-M3 | OpenAI) · store.py (Milvus: dense + BM25 sparse, repo_id partition key)
           enrich.py (isteğe bağlı LLM açıklaması, cache'li) · pipeline.py (manifest diff → chunk → embed → yaz)
search/    routing.py (sembol → BM25) · retrieve.py (kanallar → RRF → rerank, kanal skorları) · rerank.py · answer.py
db.py      SQLite: repos, files (path→sha), jobs, webhook_events, enrichment
jobs.py    tek worker kuyruğu + repo başına dedupe + Azure poller
webhooks.py Azure "Code pushed" + GitHub push (HMAC) · api.py FastAPI (+ /mcp mount) · cli.py typer
mcp_server.py MCP sunucusu (streamable HTTP): search_code · read_code · list_repos — sinyaller (zayıf eşleşme, DOKÜMAN, tazelik) · eval.py golden runner (+ negatif vakalar)
```

Veri akışı: `push → webhook/poll → job → refresh_source (fetch+reset) → dosya sha'ları
manifest ile karşılaştır → değişen dosyanın chunk'larını sil + yeniden yaz → manifest
güncelle → retriever cache'ini boşalt`.

## Bağlayıcı kararlar

- **Değişiklik tespiti içerik hash'iyle, commit diff'iyle değil.** Yeniden adlandırma,
  mod değişikliği, submodule gibi git kenar durumları yok; yerel dizinler de aynı yoldan
  geçer. Manifest satırı chunk silinmeden önce kaldırılır, yazıldıktan sonra geri konur —
  yarıda kesilen iş eksik bırakmaz.
- **Chunk = kod birimi, ≤ 2000 byte.** Büyük sınıf üyelerine bölünür, küçükler birleşir,
  yorum/import bir sonraki birime yapışır. `text` temiz kalır; başlık (dosya, sembol,
  sınıf, import'lar) yalnızca `indexed_text`'e girer.
- **Sembol biçimli sorgu → BM25, düz cümle → dense; rerank kapalı.** Ölçüldü
  (README → Ölçüm defteri): auto+dense 0.786/0.690, semboller BM25'te 1.0/1.0;
  bge-reranker-v2-m3 zarar etti (0.762/0.508, p50 2-4 sn) → varsayılan kapalı.
  Her hit kanal skorlarını taşır (`dense`, `bm25`, `rrf`, `rerank`).
- **Her retrieval değişikliği `rag eval` ile ölçülür.** "Sanki iyi oldu" sonuç değildir;
  `evals/results/` altına bir JSON düşmeden bayrak varsayılanı değişmez.
- **Enrichment (LLM açıklaması) kapalı başlar.** BGE-M3 çok dilli; Türkçe soru için
  gerekirse aç, ama önce ölç. Açıklamalar chunk hash'iyle cache'lenir, yanlış alfabe reddedilir.
- **Tek Milvus collection, `repo_id` partition key.** Şemayı değiştirmek = collection'ı
  yeniden kurmak. `RAG_*` chunk/embedding ayarı değişince `index_version` değişir ve
  bir sonraki sync tam yeniden index yapar.
- **MCP aynı süreçte, aynı retriever'ın üstünde.** `/mcp` altında streamable HTTP;
  bloklayan iş (embedding, Milvus, dosya) `anyio.to_thread` ile çalışır — MCP oturumu
  tek event loop'ta akıyor. Araç çıktısı ajana VERİ olarak işaretlenir, talimat değil.
- **Alakasızlık üç bantta ele alınır (CRAG).** kNN "yakın olan yok" demez; cosine gri
  bölgede ayırmıyor (ölçüldü, README → Çekimserlik), reranker kapısı %29 yanlış alarm
  veriyor. O yüzden: dense < 0.45 atılır (`min_dense_score`, `dropped` sayar — saçma
  kuyruk), 0.45-0.55 `weak_match` notuyla döner, üstü normal. Yanına DOKÜMAN etiketi,
  index tazeliği, manifest'e kilitli `read_code`; hakem ajan/LLM/insan. Golden'da negatif
  vakalar (`expect: []`) var; `abstain` ve `false_weak` birlikte okunur.
- **PAT hiçbir yere yazılmaz.** git'e `-c http.extraheader=` ile geçer; hata mesajları
  redakte edilir. Index'e girmeden önce `scrub` çalışır.

## Kurallar

- Türkçe yorum, İngilizce tanımlayıcı (api/ ile aynı). Yorum "neden"i anlatır.
- Modül sınırlarını geçen her şey `models.py`'de tipli; çıplak dict dolaşmaz.
- Her yeni modülün `tests/` altında karşılığı var. Milvus/model gerektiren testler
  `live` işaretli; birim testleri sahte embedder/store ile çalışır.
- Anthropic çağrıları `llm.py` üzerinden, `claude-opus-5`, streaming + `get_final_message`,
  `fallbacks="default"`. Başka yerde `anthropic.Anthropic()` açma.
- Ölçüm rakamlarını README'deki "Ölçüm defteri"ne yaz; tahmin değil, çalışmış komut çıktısı.
