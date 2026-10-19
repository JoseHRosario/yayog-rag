"""Bounded, process-private memoization of pure source derivation, never trust decisions."""

from __future__ import annotations

import copy
import hashlib
import pickle
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Concatenate, ParamSpec

from .core import _hash_file

_P = ParamSpec("_P")
_MAX_ENTRIES = 16
_MAX_BYTES = 32 * 1024 * 1024
_cache: OrderedDict[bytes, bytes] = OrderedDict()
_lock = RLock()


def memoize_derivation(
    versions: Callable[[], object],
) -> Callable[
    [Callable[Concatenate[Path, _P], list[dict[str, Any]]]],
    Callable[Concatenate[Path, _P], list[dict[str, Any]]],
]:
    """Reuse extraction only; callers must validate source/recovery/catalogs on every use.

    Keys include the actual source bytes, exact typed input snapshot and processing/
    renderer versions. Serialized values bound retained payload memory and deserialize
    to independent deep copies. Pickle bytes are created here, never read from disk or
    supplied externally. No exceptions, review states or eligibility are cached.
    """

    def decorate(
        derive: Callable[Concatenate[Path, _P], list[dict[str, Any]]],
    ) -> Callable[Concatenate[Path, _P], list[dict[str, Any]]]:
        @wraps(derive)
        def cached(source: Path, /, *args: _P.args, **kwargs: _P.kwargs) -> list[dict[str, Any]]:
            fingerprint = _hash_file(source)
            arguments, keywords = copy.deepcopy((args, kwargs))
            key = hashlib.sha256(
                pickle.dumps(
                    (
                        derive.__module__,
                        derive.__qualname__,
                        versions(),
                        str(source.resolve()),
                        fingerprint,
                        arguments,
                        keywords,
                    ),
                    protocol=5,
                )
            ).digest()
            with _lock:
                payload = _cache.get(key)
                if payload is not None:
                    _cache.move_to_end(key)
            records = (
                pickle.loads(payload)
                if payload is not None
                else derive(source, *arguments, **keywords)
            )
            # Do not publish a result under an identity that changed during extraction.
            if _hash_file(source) != fingerprint:
                raise ValueError("source_changed")
            if payload is None:
                payload = pickle.dumps(records, protocol=5)
                if len(payload) <= _MAX_BYTES:
                    with _lock:
                        _cache[key] = payload
                        _cache.move_to_end(key)
                        while (
                            len(_cache) > _MAX_ENTRIES
                            or sum(map(len, _cache.values())) > _MAX_BYTES
                        ):
                            _cache.popitem(last=False)
                return copy.deepcopy(records)
            return records

        return cached

    return decorate
