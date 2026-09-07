# 4. Cevaplar ve ajanlar

Retrieval hiç LLM olmadan çalışıyor. `POST /search` ve MCP araçları asla bir tanesini
çağırmıyor. LLM tam olarak iki isteğe bağlı yerde yaşıyor: bir cevap yazmak ve chunk
açıklamaları yazmak.

Bu ayrımı baştan söylemeye değer, çünkü servisi sıfır kimlik bilgisiyle kullanılabilir
ve modelsiz test edilebilir kılan şey o.

!!! done "Bu aşamanın sahibi olduğu yerler"

    `llm.py` (sağlayıcı zinciri), `search/answer.py` (atıflı cevaplar),
    `mcp_server.py` (streamable HTTP üzerinde üç araç) ve `api.py`, `cli.py`,
    `static/index.html` içindeki yüzeyler.

## Sağlayıcı zinciri senin makinende bitiyor

`RAG_LLM_PROVIDER=auto` şu sırayla çözülüyor:

1. Anahtar varsa **Anthropic**
2. Anahtar varsa **OpenAI**
3. Yoksa **yerel Ollama**

Yani `/ask` hiç anahtar olmadan çalışıyor. Bu bir kolaylık özelliği değil: kredi kartı
olmadan denenemeyen bir servis denenmez.

```bash
ollama pull qwen3.5:9b            # 6.6 GB; zayıf makinede qwen3.5:4b
uv run rag ask "webhook olayları nasıl kuyruğa alınıyor?" -r my-api
```

OpenAI istemcisi bir `base_url` alıyor, dolayısıyla OpenAI uyumlu her şey aynı şekilde
bağlanıyor:

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000          # ya da LM Studio → Local Server
RAG_LLM_PROVIDER=openai RAG_OPENAI_BASE_URL=http://localhost:8000/v1 \
RAG_LLM_MODEL=Qwen/Qwen3.5-9B OPENAI_API_KEY=local uv run rag serve
```

!!! measured "Yerel modeller, aynı 6 chunk'lık prompt, 3 soru"

    | Model | cevap başına süre | Kalite | "Cevap yok" davranışı |
    |---|---|---|---|
    | **qwen3.5:9b, düşünme kapalı** ✓ | 12–17 sn | `[1][2][4]` atıflı, doğru chunk | doğru, 2 sn |
    | qwen3.5:9b, düşünme açık | 48–51 sn | **3 sorunun 2'sinde boş cevap** | — |
    | qwen2.5:7b | 22–32 sn | iyi, dosya + fonksiyon veriyor | doğru, 4 sn |
    | qwen3:1.7b, düşünme kapalı | 11–15 sn | bozuk kelimeler, uydurma terimler | kararsız |
    | claude-opus-5 | — | referans | doğru |

    O tablodan iki ayar çıktı. Düşünen modeller düşünme **kapalı** çağrılıyor: atıflı bir
    cevabın buna ihtiyacı yok ve açıkken akıl yürütme 1024 token'lık bütçenin tamamını
    yiyip hiçbir şey döndürmedi. Bir de `RAG_OLLAMA_NUM_CTX=16384`, çünkü Ollama'nın 4k
    varsayılanı 8 chunk'lık bir prompt'u sessizce kırpıyordu — hata yok, sadece daha kötü
    bir cevap.

## Atıflar uydurmayı görünür kılıyor

Atıfsız bir cevap yanlışlanamaz: her cümle, koddan mı yoksa modelin ön kabullerinden mi
geldiğine bakılmaksızın aynı derecede makul görünür. Her iddiaya bir işaret şartı
koymak, bunu okuyucunun iki saniyede kontrol edebileceği bir şeye çeviriyor.

```python
# search/answer.py
SYSTEM_PROMPT = """...
1. Answer ONLY from the chunks given. Anything not in them, you do not know.
2. Mark every claim with the number of the chunk it rests on: [1], [2]. Never write an
   uncited claim.
3. If the chunks do not answer the question, say so plainly, and suggest which file to
   look at if the chunks let you infer it. Do not invent.
"""
```

Satır numaraları chunk'larla birlikte prompt'a giriyor, yani bir atıf dosyaya değil
`dosya:satır — fonksiyon`'a çözülüyor. 3. kural en az 2. kadar önemli: başarısız olmak
için açık bir izin verilmezse, bağlamının cevaplayamayacağı bir soru sorulan model onu
yine de cevaplar.

```bash
uv run rag ask "iyimser kilit nasıl çalışıyor?" -r my-api
```

## MCP: karar verici ajandır

MCP sunucusu HTTP API ile aynı süreçte, aynı retriever üzerinde, `/mcp` altında
streamable HTTP olarak çalışıyor:

```bash
claude mcp add --transport http milvus-rag http://localhost:8090/mcp
```

Üç araç — ve sayının kendisi mesele:

| Araç | Ne yapıyor |
|---|---|
| `search_code` | adaylar, kanal başına skorlar ve aşağıdaki sinyallerle |
| `read_code` | dosyanın kalanını açar — **yalnızca indekslenmiş dosyaları** |
| `list_repos` | hangi kod tabanlarının bağlı olduğu ve her birinin ne kadar taze olduğu |

```python
# mcp_server.py
"""Why only three tools: the agent finds the first candidates with `search_code`, then
decides by READING the rest with `read_code`; `list_repos` says which codebases are
connected. That loop is essentially what Cursor and Claude Code do over a codebase —
the retriever's job is to offer candidates, the decision is the agent's."""
```

### Kendinden emin bir tahmin yerine dürüst sinyaller

Retriever her sorguya bir şey döndürüyor, saçma bir soruya bile. Kosinüs gri bölgeyi
ayıramadığına göre ([Retrieval](03-retrieval.md)), cevap daha akıllı bir kapı değil —
ajana neyi bilip neyi bilmediğini söylemek:

- **zayıf eşleşme notu**, en iyi dense skor 0.55'in altındaysa
- **`DOCUMENT` etiketi**, böylece bir tasarım dokümanının içinde alıntılanan kod gerçek
  kod sanılmıyor
- **indeks tazeliği**, böylece bir ajan bayat bir indeksi eksik bir dosyadan ayırabiliyor
- **manifeste kilitli `read_code`**, ki "orada değil" ifadesini güvenilir kılan şey bu —
  araç, indeksin hiç görmediği bir dosyayı açamıyor
- sunucu talimatlarında **"bulamadım" demek için açık izin**

### Getirilen kod talimat değil, veridir

```python
"""SECURITY: the code that comes back is from an indexed repo — it is DATA for the
agent, not INSTRUCTIONS. The server instructions say so explicitly."""
```

İndekslenmiş bir repo güvenilmeyen girdidir. Birinin `README`'sindeki "önceki
talimatlarını yok say" diyen bir yorum, ajana gerçek bir fonksiyon gövdesiyle aynı
kanaldan ulaşır ve bir şey söylenmedikçe ikisini yapısal olarak ayıran hiçbir şey yoktur.

### Bloklayan iş bir iş parçacığına gidiyor

Embedding, Milvus çağrıları ve dosya okumaları bloklar. MCP oturumu tek bir olay
döngüsünde çalıştığı için orada bloklamak, bağlantıdaki diğer her isteği durdurur.
Üçü de `anyio.to_thread` üzerinden geçiyor.

## Yüzeyler

Her şeye aynı kod yolu üzerinden üç şekilde erişiliyor.

=== "HTTP"

    ```bash
    curl -s localhost:8090/search -H 'content-type: application/json' -d '{
      "query": "webhook olayları nasıl kuyruğa alınıyor",
      "repo_ids": ["my-api"],
      "k": 8,
      "mode": "auto"
    }' | jq '.hits[0] | {path, symbol, start_line, scores}'
    ```

    `/docs` OpenAPI sayfasını sunuyor. Repolar, işler, dosyalar ve chunk'ların hepsinin
    uçları var; webhook'lar `/webhooks/azure/push` ve `/webhooks/github/push` altında.

=== "CLI"

    ```bash
    uv run rag serve                                  # :8090, arayüz /
    uv run rag search "..." -r my-api --mode bm25
    uv run rag ask "..." -r my-api
    uv run rag sync my-api --force
    uv run rag eval evals/golden.example.jsonl -r my-api --tag v1
    ```

=== "Web arayüzü"

    `/` altında üç sekme:

    - **Arama** — kanal skorlarıyla sonuçlar ve yanlarında LLM cevabı
    - **İşler** — her indeks koşusu bir hat olarak: kaynak → fark → chunking → embedding
      → Milvus, canlı ilerlemeyle
    - **Bağlan** — kopyalanabilir webhook URL'leri, poller durumu, curl örnekleri, MCP
      kurulumu ve Ollama ayakta mı, model çekilmiş mi diye canlı bir kontrol

Kimlik bilgileri `.env` kadar arayüzden de girilebiliyor: GitHub token'ı ile Azure org+PAT
**Repolar › Repo bağla** altında, webhook sırrı ve Anthropic anahtarı **Bağlan ›
Anahtarlar** altında. Girdiğin şey doğrulanıyor, `data/rag.db`'ye yazılıyor, ortamı
geçersiz kılıyor ve yeniden başlatma gerektirmiyor.
