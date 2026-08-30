from __future__ import annotations

from collections.abc import Iterable, Sequence
from urllib.parse import unquote


class _RefToken:
    __slots__ = ("ref",)

    def __init__(self, ref: str) -> None:
        self.ref = ref


def sourced_refs(*groups: Iterable[object]) -> tuple[object, ...]:
    extra: list[_RefToken] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            nested = getattr(item, "analysis", None)
            refs = (
                *getattr(item, "evidence_refs", ()),
                *getattr(nested, "evidence_refs", ()),
            )
            for raw in refs:
                if isinstance(raw, str) and raw and raw not in seen:
                    seen.add(raw)
                    extra.append(_RefToken(raw))
    return tuple(extra)


def bound_cited_refs(
    cited: Sequence[str], items: Iterable[object]
) -> tuple[str, ...]:
    allowed = {
        ref: item
        for item in items
        if isinstance((ref := getattr(item, "ref", None)), str) and ref
    }
    resolved: list[str] = []
    for raw in cited:
        if not isinstance(raw, str) or not raw.strip():
            continue
        match = _bound_ref(raw.strip(), allowed)
        if match is not None and match not in resolved:
            resolved.append(match)
    return tuple(resolved)


def _bound_ref(cited: str, allowed: dict[str, object]) -> str | None:
    if cited in allowed:
        return cited
    cited_norm = unquote(cited)
    cited_url = cited_norm.removeprefix("gdelt:")
    for ref, item in allowed.items():
        url = getattr(item, "url", "") or ""
        if cited_norm == unquote(ref):
            return ref
        if cited_url == url or cited_url == unquote(ref).removeprefix("gdelt:"):
            return ref
        if url and (cited_url.endswith(url) or url.endswith(cited_url)):
            return ref
        if cited_norm.endswith(ref) or ref.endswith(cited):
            return ref
    return None
