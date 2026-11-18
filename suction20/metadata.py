"""Small structural metadata helpers; no content checksums or file reads."""
from __future__ import annotations


def without_legacy_checksums(value):
    """Discard historical checksum declarations when publishing new records.

    Old manifests/configs remain readable. These fields no longer constitute a
    validation contract; shape, configuration and transaction IDs are checked by
    the callers. Tensor/array objects are returned unchanged.
    """
    if isinstance(value, dict):
        return {key: without_legacy_checksums(item) for key, item in value.items()
                if 'sha256' not in str(key).lower().replace('_', '').replace('-', '')}
    if isinstance(value, list):
        return [without_legacy_checksums(item) for item in value]
    if isinstance(value, tuple):
        return tuple(without_legacy_checksums(item) for item in value)
    return value
