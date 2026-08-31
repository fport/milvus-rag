# Kendi sunucunda çalıştırma

İki yol var: **Docker Compose** (önerilen — Milvus zaten konteyner istiyor) ya da
Milvus'u konteynerde tutup servisi **systemd** ile çıplak çalıştırmak.

## Gereksinimler

| Kaynak | En az | Not |
|---|---|---|
| İşletim sistemi | Linux x86_64 / arm64 | Docker 24+, compose v2 |
| RAM | 8 GB (rahatı 16 GB) | bge-m3 ~3-4 GB, Milvus ~2-3 GB |
| Disk | 20 GB+ | model cache ~5 GB + Milvus verisi + repo klonları |
| CPU | 4+ çekirdek | embedding CPU'da ~5-8 chunk/sn; 3k chunk'lık repo ~8-10 dk |
| GPU | gerekmez | varsa `RAG_EMBEDDING_DEVICE=cuda` ile 5-10x hız |

Not: CPU yavaş geliyorsa `RAG_EMBEDDING_BACKEND=openai` (text-embedding-3-large,
`OPENAI_API_KEY` gerekir) indexlemeyi API'ye taşır; model indirmesi de gerekmez.

## Yol 1 — Docker Compose (önerilen)

```bash
git clone <bu-repo> && cd <repo>
cp .env.example .env        # AZURE_DEVOPS_* / GITHUB_TOKEN / RAG_WEBHOOK_SECRET / ANTHROPIC_API_KEY
docker compose -f infra/docker-compose.prod.yml up -d --build
docker compose -f infra/docker-compose.prod.yml logs -f rag   # "hazır" satırını bekle
curl -f http://localhost:8090/health
```

- İlk açılışta embedding modeli (~2.2 GB) Hugging Face'ten iner; `hf-cache`
  volume'ünde kalır, sonraki açılışlar saniyeler sürer.
- `rag-data` volume'ü SQLite'ı (repo kayıtları, manifest, işler) ve repo
  klonlarını tutar. **Yedeklenecek tek şey budur** — Milvus'taki veri türevdir,
  `rag sync --force` ile yeniden üretilir.
- Güncelleme: `git pull && docker compose -f infra/docker-compose.prod.yml up -d --build`
  (chunk/embedding ayarı değiştiyse servis bir sonraki sync'te otomatik tam
  yeniden index yapar).

## Yol 2 — systemd (Milvus konteynerde, servis çıplak)

```bash
docker compose -f infra/docker-compose.yml up -d      # yalnız Milvus + etcd + MinIO
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --no-dev
```

`/etc/systemd/system/milvus-rag.service`:

```ini
[Unit]
Description=Milvus RAG
After=network-online.target docker.service

[Service]
User=rag
WorkingDirectory=/opt/milvus-rag
ExecStart=/home/rag/.local/bin/uv run rag serve
Restart=on-failure
EnvironmentFile=/opt/milvus-rag/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now milvus-rag
```

## Dışa açma: reverse proxy + TLS

**Servisin kendi kimlik doğrulaması yok.** 8090'ı doğrudan internete açma:

- Webhook'lar dışarıdan erişilebilir olmalı (Azure/GitHub push gönderecek),
- geri kalan her şey (arama, repo ekleme, dosya okuma!) iç ağda kalmalı.

Caddy ile pratik ayrım (otomatik TLS dahil) — `/etc/caddy/Caddyfile`:

```caddy
rag.example.com {
    # Webhook uçları herkese açık; güvenlik zaten imza/sır ile:
    # Azure → paylaşılan sır, GitHub → HMAC-SHA256 (RAG_WEBHOOK_SECRET).
    handle /webhooks/* {
        reverse_proxy localhost:8090
    }
    # Geri kalanı basic auth arkasında (şifre üret: caddy hash-password)
    handle {
        basic_auth {
            admin <bcrypt-hash>
        }
        reverse_proxy localhost:8090
    }
}
```

nginx kullanıyorsan aynı ayrım `location /webhooks/ { proxy_pass ... }` + geri
kalana `auth_basic` ile kurulur. Alternatif: servis hiç dışa açılmaz, webhook
yerine yalnızca poller kullanılır (`RAG_POLL_INTERVAL_SECONDS`, varsayılan 300 sn)
— push'tan en geç 5 dk sonra index tazelenir, hiçbir portu açman gerekmez.

## Webhook kurulumları

- **Azure DevOps:** Project Settings → Service Hooks → Web Hooks → *Code pushed*
  → URL `https://rag.example.com/webhooks/azure/push`, Basic auth password (ya da
  `X-RAG-Webhook-Secret` header'ı) = `RAG_WEBHOOK_SECRET`.
- **GitHub:** repo → Settings → Webhooks → Add webhook → URL
  `https://rag.example.com/webhooks/github/push`, content type
  `application/json`, Secret = `RAG_WEBHOOK_SECRET`, "Just the push event".
  (GitHub "Recent Deliveries" sekmesinden teslimatları ve cevapları görürsün;
  ping olayına servis `{"pong": true}` döner.)

## Sağlık ve işletme

| Ne | Nasıl |
|---|---|
| Sağlık | `GET /health` — Milvus bağlantısı, model, repo sayısı |
| İş takibi | `GET /jobs?repo_id=...` — durum + `stats.progress` |
| Loglar | `docker compose ... logs -f rag` (tek satır, `anahtar=değer`) |
| Milvus arayüzü | `--profile ui` ile Attu (:8091) — üretimde kapalı tut |
| Tam yeniden index | `POST /repos/{id}/sync {"force": true}` |

Bilinen sınırlar: tek worker (indexleme sıralı — kasıtlı, bkz. CLAUDE.md);
Milvus standalone tek düğüm. İkisi de tek sunucu kurulumu için yeterli.
