"""Canonical speaker labels at the speech-to-ERC boundary."""

from __future__ import annotations


def normalize_speaker_id(
    speaker_id: str, *, unknown_speaker_id: str = "UNKNOWN"
) -> str:
    """Trim IDs and canonicalize unknown labels without merging known identities."""
    speaker = speaker_id.strip()
    unknown = unknown_speaker_id.strip()
    if not speaker:
        raise ValueError("speaker_id must not be empty")
    if not unknown:
        raise ValueError("unknown_speaker_id must not be empty")
    if speaker.casefold() in ("unknown", unknown.casefold()):
        return unknown
    return speaker
