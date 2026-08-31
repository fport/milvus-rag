"""Sorgunun biçiminden retriever seçimi.

production-ready-rag-system'de ölçüldü (evals/retrieval.yaml, 30 soru):

| recall@5      | sembol araması | düz cümle |
|---------------|----------------|-----------|
| dense         | 0.60           | 0.62      |
| BM25          | 0.80           | 0.36      |
| RRF (ikisi)   | 0.60           | 0.52      |

Sembol biçimli sorgu (`handleAuthCallback`, `QUEUE_NAMES`, `auth.service`) için
embedding'in söyleyecek bir şeyi yok; BM25'in var. Düz cümle için tersi. İkisini
her zaman birleştirmek, bulamayan kanalın da en iyi tahminini tam güçle terfi
ettirir. Yönlendirme bir regex'e mal olur.
"""

import re

_SYMBOL_SHAPED = re.compile(r"^[A-Za-z_][\w.\-/:]*$")
_CAMEL_BOUNDARY = re.compile(r"[a-z0-9][A-Z]")


def looks_like_symbol(query: str) -> bool:
    """Tek token, içinde bir sınır var: camelCase, snake_case, kebab-case, a.b.c, a/b.

    Bilerek dar: yanlış pozitif gerçek bir soruyu BM25'e gönderir (düz cümlede
    ölçülebilir kötü); yanlış negatif yalnızca bir iyileşmeden vazgeçer.
    """
    text = query.strip()
    if not text or any(character.isspace() for character in text):
        return False
    if not _SYMBOL_SHAPED.match(text):
        return False
    return bool(
        _CAMEL_BOUNDARY.search(text)
        or "_" in text
        or "-" in text
        or "." in text
        or "/" in text
        or "::" in text
        or (text.isupper() and len(text) > 1)
    )
