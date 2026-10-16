"""Align target-turn transcripts without consulting timestamps or emotion labels.

``align_video`` accepts one video's CSV rows in transcript order. The three
reference turns participate in one monotonic, word-level edit alignment; only
turns 1 and 3 produce evaluation records. Thus the phone response anchors the
alignment without becoming an evaluation target. Token spans are zero-based,
half-open, and local to the corresponding reference/prediction row. Counts use
normalized tokens (contractions expand, and common number words become digits).

No semantic matching is used: invented speech must not become evidence for an
emotion prediction. Low-evidence overlaps are retained as unmatched targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
import unicodedata


_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
_NUMBER_WORDS = dict(
    zip(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        map(str, range(21)),
    )
)
_CONTRACTIONS = {
    "won't": ("will", "not"),
    "can't": ("can", "not"),
    "cannot": ("can", "not"),
    "shan't": ("shall", "not"),
    "let's": ("let", "us"),
    "alright": ("all", "right"),
}
_STOPWORDS = frozenset(
    "a an the i me my you your he him his she her it its we us our they them "
    "their is am are was were be been being have has had do does did and or but "
    "of to in on at for from with by as that this these those there here so "
    "just oh uh um well then than if not will would can could should all".split()
)


@dataclass(frozen=True)
class _Token:
    value: str
    row: int
    local: int
    start: int
    end: int


def _normal_words(word: str) -> tuple[str, ...]:
    if word in _CONTRACTIONS:
        return _CONTRACTIONS[word]
    if word.endswith("n't"):
        return (word[:-3], "not")
    for suffix, expansion in (
        ("'re", "are"),
        ("'ve", "have"),
        ("'ll", "will"),
        ("'m", "am"),
        ("'d", "would"),
        ("'s", "is"),
    ):
        if word.endswith(suffix):
            return (word[: -len(suffix)], expansion)
    return (_NUMBER_WORDS.get(word, word),)


def _tokens(rows: list[dict]) -> tuple[list[_Token], list[list[_Token]]]:
    result: list[_Token] = []
    per_row: list[list[_Token]] = []
    for row_index, row in enumerate(rows):
        text = str(row.get("text") or "")
        # Curly apostrophe replacement preserves original character offsets.
        comparable = text.replace("\u2019", "'").replace("\u2018", "'")
        row_tokens = []
        for match in _WORD.finditer(comparable):
            word = unicodedata.normalize("NFKC", match.group()).casefold()
            for value in _normal_words(word):
                token = _Token(value, row_index, len(row_tokens), *match.span())
                row_tokens.append(token)
                result.append(token)
        per_row.append(row_tokens)
    return result, per_row


def normalized_tokens(text: str) -> list[str]:
    """Return the normalized words used for alignment and reported counts."""
    return [token.value for token in _tokens([{"text": text}])[0]]


def _similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if min(len(left), len(right)) < 4 or left.isdigit() or right.isdigit():
        return 0.0
    similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
    return similarity if similarity >= 0.8 else 0.0


def _edit_alignment(reference: list[_Token], prediction: list[_Token]):
    """Return supported pairs and equally optimal alternative assignments.

    An unrelated substitution costs 1.25, a missing/extra word costs 1, and a
    fuzzy spelling match costs 0.35--0.8. Substitutions can establish edit
    position, but only exact/fuzzy lexical matches count as matched evidence.
    """
    nr, np = len(reference), len(prediction)
    similarities = [
        [_similarity(ref.value, pred.value) for pred in prediction]
        for ref in reference
    ]
    costs = [
        [0.0 if s == 1 else 0.35 + 2.25 * (1 - s) if s else 1.25 for s in row]
        for row in similarities
    ]
    forward = [[0.0] * (np + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        forward[i][0] = float(i)
    for j in range(np + 1):
        forward[0][j] = float(j)
    for i in range(1, nr + 1):
        for j in range(1, np + 1):
            forward[i][j] = min(
                forward[i - 1][j - 1] + costs[i - 1][j - 1],
                forward[i - 1][j] + 1,
                forward[i][j - 1] + 1,
            )
    backward = [[0.0] * (np + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        backward[i][np] = float(nr - i)
    for j in range(np + 1):
        backward[nr][j] = float(np - j)
    for i in range(nr - 1, -1, -1):
        for j in range(np - 1, -1, -1):
            backward[i][j] = min(
                backward[i + 1][j + 1] + costs[i][j],
                backward[i + 1][j] + 1,
                backward[i][j + 1] + 1,
            )
    optimum = forward[nr][np]
    possible_predictions: dict[int, set[int]] = {}
    possible_references: dict[int, set[int]] = {}
    for i in range(nr):
        for j in range(np):
            if similarities[i][j] and math.isclose(
                forward[i][j] + costs[i][j] + backward[i + 1][j + 1],
                optimum,
                abs_tol=1e-8,
            ):
                possible_predictions.setdefault(i, set()).add(j)
                possible_references.setdefault(j, set()).add(i)
    pairs = []
    i, j = nr, np
    while i or j:
        if i and j and math.isclose(
            forward[i][j],
            forward[i - 1][j - 1] + costs[i - 1][j - 1],
            abs_tol=1e-8,
        ):
            if similarities[i - 1][j - 1]:
                pairs.append((i - 1, j - 1, similarities[i - 1][j - 1]))
            i -= 1
            j -= 1
        elif i and math.isclose(
            forward[i][j], forward[i - 1][j] + 1, abs_tol=1e-8
        ):
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs)), possible_predictions, possible_references


def _basename(value: str) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _validate_video(reference_rows: list[dict], prediction_rows: list[dict]):
    if len(reference_rows) != 3:
        raise ValueError("Exactly three ordered ground-truth turns are required")
    names = {
        _basename(row["video_file"])
        for row in reference_rows + prediction_rows
        if row.get("video_file")
    }
    if len(names) > 1:
        raise ValueError("Filter by video filename before calling align_video")


def _fragment(text: str, tokens: list[_Token]) -> str:
    return text[tokens[0].start : tokens[-1].end] if tokens else ""


def _informative(token: _Token) -> bool:
    return token.value not in _STOPWORDS and len(token.value) >= 3


def _accepted_target(pairs, reference, prediction, count):
    if not pairs or not count:
        return False
    coverage = len(pairs) / count
    informative = sum(_informative(reference[i]) for i, _, _ in pairs)
    prediction_span = pairs[-1][1] - pairs[0][1] + 1
    quality = sum(s for _, _, s in pairs)
    similarity = 2 * quality / (count + prediction_span)
    if count <= 2 and coverage >= 0.5:
        # A short complete response ("yes", "thank you") is still meaningful.
        return quality / len(pairs) >= 0.9 and (
            coverage == 1 or informative >= 1 or count == 1
        )
    if len(pairs) < 2 or not informative:
        return False
    if coverage >= 0.3 and similarity >= 0.35:
        return True
    # Permit a short, consecutive content phrase from a much longer turn,
    # retaining its small coverage instead of pretending the turn was complete.
    consecutive = any(
        b[0] == a[0] + 1 and b[1] == a[1] + 1
        for a, b in zip(pairs, pairs[1:])
    )
    return (
        coverage >= 0.15
        and informative >= 2
        and quality / prediction_span >= 0.75
        and consecutive
    )


def align_video(reference_rows: list[dict], prediction_rows: list[dict]) -> list[dict]:
    """Return one record per meaningful target-turn/prediction overlap.

    Rows must already be filtered by filename and ordered as in the CSV. All
    matching uses only ``text``. ``video_file`` is checked to prevent accidental
    cross-video alignment. The reference must contain exactly three turns.

    Each target without accepted lexical evidence produces one ``unmatched``
    record with ``prediction_segment=None``. ``reference_coverage`` summarizes
    all accepted overlaps for that target; ``segment_reference_coverage`` is
    the current overlap's share. ``text_similarity`` is a token Dice similarity
    of the matched fragments, with fuzzy spelling matches weighted by their
    character similarity. ``prediction_tokens`` counts the full segment, so a
    merged target/phone segment remains identifiable.

    ``alignment_status`` is ``matched``, ``partial``, ``ambiguous``, or
    ``unmatched``. Ambiguity means equally optimal text assignments can cross
    reference-turn or prediction-segment boundaries. Known merged turns are a
    separate ``segmentation_status`` and do not make a clear text match
    ambiguous. Character spans and token spans are zero-based and half-open.
    """
    _validate_video(reference_rows, prediction_rows)
    reference, ref_per_row = _tokens(reference_rows)
    prediction, pred_per_row = _tokens(prediction_rows)
    pairs, possible_preds, possible_refs = _edit_alignment(reference, prediction)
    turns_by_prediction: dict[int, set[int]] = {}
    for i, j, _ in pairs:
        turns_by_prediction.setdefault(prediction[j].row, set()).add(reference[i].row + 1)

    output = []
    for target in (0, 2):
        ref_text = str(reference_rows[target].get("text") or "")
        target_pairs = [pair for pair in pairs if reference[pair[0]].row == target]
        count = len(ref_per_row[target])
        accepted = _accepted_target(target_pairs, reference, prediction, count)
        groups: dict[int, list] = {}
        if accepted:
            for pair in target_pairs:
                groups.setdefault(prediction[pair[1]].row, []).append(pair)
            # One generic shared word in a long unrelated segment is too weak
            # to score. Single-word split segments remain valid when adjacent
            # to other aligned words from the same target.
            for segment, group in list(groups.items()):
                if len(group) != 1 or count <= 2:
                    continue
                i, j, _ = group[0]
                adjacent = any(
                    abs(i - other_i) == 1 and abs(j - other_j) == 1
                    for other_i, other_j, _ in target_pairs
                )
                if not (
                    (len(pred_per_row[segment]) == 1 and adjacent)
                    or (_informative(reference[i]) and adjacent)
                ):
                    del groups[segment]
        accepted_pairs = [pair for group in groups.values() for pair in group]
        coverage = len(accepted_pairs) / count if count else 0.0
        possible_segments = sorted(
            {
                prediction[j].row + 1
                for i, token in enumerate(reference)
                if token.row == target
                for j in possible_preds.get(i, ())
            }
        )
        base = {
            "reference_turn": target + 1,
            "reference_text": ref_text,
            "reference_tokens": count,
            "reference_coverage": coverage,
            "candidate_prediction_segments": possible_segments,
        }
        if not groups:
            output.append(
                {
                    **base,
                    "prediction_segment": None,
                    "prediction_text": "",
                    "matched_reference_text": "",
                    "matched_prediction_text": "",
                    "matched_tokens": 0,
                    "prediction_tokens": 0,
                    "segment_reference_coverage": 0.0,
                    "text_similarity": 0.0,
                    "alignment_status": "unmatched",
                    "ambiguity_reasons": "",
                    "segmentation_status": "unmatched",
                    "prediction_reference_turns": [],
                    "prediction_spans_phone_response": False,
                    "reference_token_start": None,
                    "reference_token_end": None,
                    "prediction_token_start": None,
                    "prediction_token_end": None,
                    "reference_char_start": None,
                    "reference_char_end": None,
                    "prediction_char_start": None,
                    "prediction_char_end": None,
                    "matched_reference_token_indices": [],
                    "matched_prediction_token_indices": [],
                }
            )
            continue
        for segment, group in groups.items():
            ref_tokens = [reference[i] for i, _, _ in group]
            pred_tokens = [prediction[j] for _, j, _ in group]
            pred_text = str(prediction_rows[segment].get("text") or "")
            reasons = set()
            for i, j, _ in group:
                if len({prediction[k].row for k in possible_preds.get(i, ())}) > 1:
                    reasons.add("alternative_prediction_segments")
                if len({reference[k].row for k in possible_refs.get(j, ())}) > 1:
                    reasons.add("alternative_reference_turns")
            ref_span = ref_tokens[-1].local - ref_tokens[0].local + 1
            pred_span = pred_tokens[-1].local - pred_tokens[0].local + 1
            similarity = 2 * sum(s for _, _, s in group) / (ref_span + pred_span)
            turns = sorted(turns_by_prediction.get(segment, set()))
            output.append(
                {
                    **base,
                    "prediction_segment": segment + 1,
                    "prediction_text": pred_text,
                    "matched_reference_text": _fragment(ref_text, ref_tokens),
                    "matched_prediction_text": _fragment(pred_text, pred_tokens),
                    "matched_tokens": len(group),
                    "prediction_tokens": len(pred_per_row[segment]),
                    "segment_reference_coverage": len(group) / count,
                    "text_similarity": similarity,
                    "alignment_status": (
                        "ambiguous" if reasons else "matched" if coverage >= 1 else "partial"
                    ),
                    "ambiguity_reasons": ";".join(sorted(reasons)),
                    "segmentation_status": "merged_turns" if len(turns) > 1 else "single_turn",
                    "prediction_reference_turns": turns,
                    "prediction_spans_phone_response": 2 in turns,
                    "reference_token_start": ref_tokens[0].local,
                    "reference_token_end": ref_tokens[-1].local + 1,
                    "prediction_token_start": pred_tokens[0].local,
                    "prediction_token_end": pred_tokens[-1].local + 1,
                    "reference_char_start": ref_tokens[0].start,
                    "reference_char_end": ref_tokens[-1].end,
                    "prediction_char_start": pred_tokens[0].start,
                    "prediction_char_end": pred_tokens[-1].end,
                    "matched_reference_token_indices": [token.local for token in ref_tokens],
                    "matched_prediction_token_indices": [token.local for token in pred_tokens],
                }
            )
    return output
