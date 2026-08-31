"""İsteğe bağlı: her chunk için LLM'den 2-3 cümlelik Türkçe açıklama.

Bir kod parçasında doğal dil yok denecek kadar az; olanı da yazarının yorum
dilinde. production-ready-rag-system'de ölçüldü: Türkçe düz sorular, İngilizce
kod üstünde recall@5 = 0.04. Çare daha iyi eşleştirici değil, eksik metni yazmak
(Anthropic "contextual retrieval": retrieval hatasında %35 düşüş).

BGE-M3 çok dilli olduğu için bu katman varsayılan olarak kapalı; açmadan önce
`rag eval` ile ölç. Açıksa açıklamalar chunk hash'iyle cache'lenir: yeniden
index aynı parayı ikinci kez ödemez.

Dosya başına tek çağrı (dosyanın tamamı + numaralı parçalar). Yanıt kapsama
kontrolünden geçer: 7B bir model tek nesne döndürüp gerisini sessizce atabilir.
Türkçe olmayan alfabe (CJK, Hangul, Kiril) reddedilir — qwen2.5:7b yük altında
cümle ortasında Çince'ye kayıyor ve bunu söylemiyor.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

from milvus_rag.db import Database
from milvus_rag.llm import LLM, LLMError
from milvus_rag.log import get_logger
from milvus_rag.models import ChunkRecord

log = get_logger("enrich")

FILE_CONTEXT_CHARS = 8_000

SYSTEM_PROMPT = """Sen bir kod tabanını Türkçe belgeleyen teknik yazarsın.
Sana bir dosyanın tamamı ve o dosyadan numaralandırılmış parçalar verilecek.
HER PARÇA için, o parçadaki kodun ne yaptığını 2-3 cümleyle Türkçe anlat.

Şu JSON şemasında yanıt ver, başka hiçbir şey yazma:
{"parcalar": [{"n": <parça numarası>, "aciklama": "<2-3 cümle Türkçe>"}]}

Zorunlu kurallar:
- VERİLEN HER parça için bir girdi yaz. Hiçbirini atlama.
- Yalnızca ilgili parçadaki kodu anlat, dosyanın geri kalanını değil.
- Fonksiyon, değişken, tip ve tablo isimlerini ÇEVİRME, İngilizce aynen yaz.
- "Bu parça", "bu dosya" diye başlama; doğrudan kodun ne yaptığını yaz.
- Parçada geçmeyen bir fonksiyondan bahsetme. Emin değilsen kısa yaz, uydurma."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_WRONG_SCRIPT = re.compile(r"[぀-ヿ一-鿿가-힯Ѐ-ӿ]")


def chunk_hash(record: ChunkRecord) -> str:
    return hashlib.sha256(f"{record.header}\n{record.text}".encode()).hexdigest()


def is_usable(description: str) -> bool:
    text = description.strip()
    return len(text) >= 20 and not _WRONG_SCRIPT.search(text)


class Enricher:
    def __init__(self, llm: LLM, db: Database, batch_chunks: int = 6) -> None:
        self.llm = llm
        self.db = db
        self.batch_chunks = batch_chunks
        self.generated = 0
        self.cached = 0
        self.rejected = 0
        self.failures = 0

    @property
    def model(self) -> str:
        return self.llm.model

    def describe(self, path: str, text: str, records: Sequence[ChunkRecord]) -> list[str]:
        """Kayıt başına bir açıklama; üretilemeyen için boş string."""
        if not records:
            return []
        hashes = [chunk_hash(record) for record in records]
        known = self.db.get_enrichments(hashes, self.model)
        self.cached += len(known)
        descriptions = [known.get(digest, "") for digest in hashes]
        missing = [index for index, digest in enumerate(hashes) if digest not in known]
        if not missing:
            return descriptions

        fresh: list[tuple[str, str]] = []
        for start in range(0, len(missing), self.batch_chunks):
            batch = missing[start : start + self.batch_chunks]
            answers = self._describe_batch(path, text, records, batch)
            for index, description in zip(batch, answers, strict=True):
                if description:
                    descriptions[index] = description
                    fresh.append((hashes[index], description))
        if fresh:
            self.db.set_enrichments(fresh, self.model)
        return descriptions

    def _describe_batch(
        self, path: str, text: str, records: Sequence[ChunkRecord], indices: list[int]
    ) -> list[str]:
        parts = "\n\n".join(
            f"### Parça {number}\n{records[index].text}" for number, index in enumerate(indices, 1)
        )
        user = (
            f"Dosya: {path}\n\n```\n{text[:FILE_CONTEXT_CHARS]}\n```\n\n"
            f"Numaralandırılmış parçalar:\n\n{parts}"
        )
        try:
            raw = self.llm.complete(SYSTEM_PROMPT, user, max_tokens=2048)
        except LLMError as error:
            self.failures += len(indices)
            log.warning("enrichment çağrısı başarısız", path=path, error=str(error))
            return [""] * len(indices)

        parsed = _parse(raw)
        results: list[str] = []
        for number, _ in enumerate(indices, 1):
            description = parsed.get(number, "").strip()
            if description and is_usable(description):
                results.append(description)
                self.generated += 1
            else:
                results.append("")
                if description:
                    self.rejected += 1
                else:
                    self.failures += 1
        return results


def _parse(raw: str) -> dict[int, str]:
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    items = payload.get("parcalar") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return {}
    found: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_number = item.get("n")
        if raw_number is None:
            continue
        try:
            number = int(raw_number)
        except (TypeError, ValueError):
            continue
        found[number] = str(item.get("aciklama") or "")
    return found
