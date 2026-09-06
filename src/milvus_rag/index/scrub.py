"""Index'e girmeden önce sır ve kişisel veri temizliği.

Vektör geri döndürülemez ama yanındaki `content` alanı aynen saklanır ve her
atıfta prompt'a geri gelir. Bir token bir kez indexlenince cache'te, logda ve
modelin cevabında dolaşır.

Kurallar bilerek tutucu. Gerçek bir repoya karşı ölçüldü: yalnızca isme bakan
kural 78 dosyada 247 değeri kararttı ve neredeyse hiçbiri sır değildi
(`token: text(` bir DB kolonu, `secret: string` bir tip).
Bu sürüm aynı repoda 2 karartma yaptı, biri gerçek token.
"""

import re
from dataclasses import dataclass

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("[API_KEY]", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}\b")),
    ("[API_KEY]", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("[API_KEY]", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("[API_KEY]", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # Azure DevOps PAT: 52 karakter base32/ base64 karışımı, çoğunlukla küçük harf+rakam.
    ("[API_KEY]", re.compile(r"\b[a-z0-9]{52}\b")),
    ("[JWT]", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")),
    ("[EMAIL]", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("[PHONE]", re.compile(r"\+\d{1,3}[\s.\-]?\(?\d{2,4}\)?[\s.\-]?\d{3}[\s.\-]?\d{2,4}\b")),
    ("[PHONE]", re.compile(r"\b\d{3}[.\-]\d{3}[.\-]\d{4}\b")),
)

# SCREAMING_SNAKE anahtara atanmış, kendi başına şekli olmayan değer:
# DB_PASSWORD=hunter2 sırdır; `promptTokens` ya da `this.accessToken` değildir.
_ASSIGNED_SECRET = re.compile(
    r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|APIKEY|API_KEY|PRIVATE_KEY|PAT)[A-Z0-9_]*"
    r"|(?:PASSWORD|SECRET|TOKEN|APIKEY|API_KEY|PRIVATE_KEY)[A-Z0-9_]*)"
    r"(\s*[:=]\s*)"
    r"(\"[^\"\n]*\"|\'[^\'\n]*\'|[^\s\"\'#,;]+)"
)

_NOT_A_SECRET = (
    re.compile(r"(?i)\b(?:process\.env|import\.meta\.env|os\.environ|env|Environment)\b"),
    re.compile(r"^\$\{?"),
    re.compile(r"^(?:string|number|boolean|text|varchar|uuid|jsonb?|any|unknown)\b"),
    re.compile(r"[.(]\s*$"),
    re.compile(r"^[A-Za-z_][\w.]*\("),
    re.compile(r"^(?:this|self)\."),
    re.compile(r"^(?:https?|wss?|postgres(?:ql)?|redis|mongodb)://"),
    re.compile(r"(?i)(?:^<|your[-_]|change[-_]?me|placeholder|example|xxx+|\.\.\.)"),
)

MINIMUM_SECRET_LENGTH = 12

_RESERVED_DOMAINS = re.compile(
    r"(?i)@(?:example\.(?:com|net|org)|localhost|[\w.\-]+\.(?:test|invalid|example|local))$"
)


@dataclass(frozen=True, slots=True)
class ScrubResult:
    text: str
    redactions: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.redactions.values())


def _is_secret_value(value: str) -> bool:
    stripped = value.strip().strip("\"'").strip()
    if len(stripped) < MINIMUM_SECRET_LENGTH:
        return False
    if any(character.isspace() for character in stripped):
        return False
    if not (any(c.isdigit() for c in stripped) and any(c.isalpha() for c in stripped)):
        return False
    return not any(pattern.search(stripped) for pattern in _NOT_A_SECRET)


def scrub(text: str) -> ScrubResult:
    redactions: dict[str, int] = {}

    def count(label: str, amount: int) -> None:
        if amount:
            redactions[label] = redactions.get(label, 0) + amount

    cleaned = text
    for label, pattern in _PATTERNS:
        if label == "[EMAIL]":
            cleaned, replaced = _redact_emails(cleaned, pattern)
        else:
            cleaned, replaced = pattern.subn(label, cleaned)
        count(label, replaced)

    replaced = 0

    def redact_assignment(match: re.Match[str]) -> str:
        nonlocal replaced
        if not _is_secret_value(match.group(3)):
            return match.group(0)
        replaced += 1
        return f"{match.group(1)}{match.group(2)}[SECRET]"

    cleaned = _ASSIGNED_SECRET.sub(redact_assignment, cleaned)
    count("[SECRET]", replaced)
    return ScrubResult(text=cleaned, redactions=redactions)


def _redact_emails(text: str, pattern: re.Pattern[str]) -> tuple[str, int]:
    replaced = 0

    def redact(match: re.Match[str]) -> str:
        nonlocal replaced
        if _RESERVED_DOMAINS.search(match.group(0)):
            return match.group(0)
        replaced += 1
        return "[EMAIL]"

    return pattern.sub(redact, text), replaced
