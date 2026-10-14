# AIMARA integration

`prepare_manifest.py` converts the synthetic-video reference annotations into a
manifest shared by the baseline and ROB speech-enhanced pipelines. It accepts
both comma- and semicolon-delimited AIMARA CSV files, references the original
video for visual extraction, and creates mono 16 kHz WAV files per utterance.

From the repository root, prepare the clean `Anger.mp4` example with:

```bash
python -m aimara.prepare_manifest \
  --input-csv /home/Imatge/media/ssd2/AIMARA/input_csv/selfie-videos.csv \
  --video-dir /home/Imatge/media/ssd2/AIMARA/videos/selfie-videos \
  --output-dir /tmp/aimara-baseline-anger \
  --video Anger.mp4
```

The command writes `manifest.csv` and utterance WAV files under `audio/`.
Use `--no-audio` to validate and convert annotations without running FFmpeg.
The manifest deliberately omits MELD's optional `Sentiment` field to avoid
deriving an input feature from the ground-truth emotion label.

## ROB speech dependencies

The VAD adapter uses PyTorch and the Silero model loaded through Torch Hub. The
ASR adapter uses the same Faster-Whisper version pinned by the ROB repository:

```bash
python -m pip install faster-whisper==1.2.1
```

Model loading is explicit so importing `aimara` does not initialize or download
speech models:

```python
from aimara.rob import ROBTranscriber, ROBVoiceActivityDetector

vad = ROBVoiceActivityDetector.from_torch_hub(
    weights_dir="/path/to/vad-weights",
)
asr = ROBTranscriber.from_faster_whisper(
    model_size="turbo",
    weights_dir="/path/to/asr-weights",
    compute_type="float16",
    language="es,ca",
)
```

ASR requests word timestamps and returns an immutable `Transcript` containing
joined `text`, `language`, timed `segments`, and a flattened `words` property.
Each `TranscriptSegment` and `TranscriptWord` has `start_time`, `end_time`, and
`text`. Times are estimated seconds relative to the audio passed to ASR, including
its onset pre-buffer. Add the VAD region's start time to map them to the source
video. Word text retains Whisper's leading spaces and punctuation.

```python
transcript = asr.transcribe(segment_audio, sample_rate=16000)
for word in transcript.words:
    print(word.start_time, word.end_time, word.text)
```

Segment boundaries are retained even when a backend returns no word timings.
`Transcript(text, language)` remains valid for text-only adapters and test doubles.
With diarization enabled, the offline processor groups these timings by speaker
and produces separate ERC inputs for speaker changes within a VAD region.

The models are composed by the offline processor, which writes a WAV for each
returned `SpeechSegment`. With diarization disabled it returns one segment per
VAD region; with diarization enabled a region can yield several speaker groups:

```python
from aimara import OfflineROBProcessor, OfflineVADProcessor, ROBProcessorConfig

config = ROBProcessorConfig()
processor = OfflineROBProcessor(
    vad_processor=OfflineVADProcessor(vad, config),
    transcriber=asr,
)
segments = processor.process_video(
    "/path/to/video.mp4",
    "/path/to/output",
)
```

Without a diarizer, speaker IDs default to `UNKNOWN`. Optional offline
ReDimNet2 diarization assigns session-local `speaker1`, `speaker2`, etc.:

```python
from aimara.rob import ROBSpeakerDiarizer

diarizer = ROBSpeakerDiarizer.from_torch_hub(
    weights_dir="/path/to/diarization-weights",
    device="cpu",
)
processor = OfflineROBProcessor(
    vad_processor=OfflineVADProcessor(vad, config),
    transcriber=asr,
    diarizer=diarizer,
)
```

The adapter loads the new ROB repository's pinned ReDimNet2 `b6/lm` checkpoint
trained on `vb2+vox2+cnc2_v0`. It uses the vendored ROB voice identity matcher
with its upstream license and provenance under `aimara/rob/third_party/`;
neither ROS nor MongoDB is required. The model stays loaded across batch videos,
while the identity gallery resets for each recording. IDs are therefore local
to each video rather than persistent person identities.

ReDimNet2's [pinned requirements](https://github.com/PalabraAI/redimnet2/blob/c5bbe0b76e37df698c403f8844e41304ceab6307/requirements.txt)
are PyTorch, Torchaudio, NumPy, and SciPy. They are already present in
`aimara_env`; this adapter does not require NeMo or the ROS speech stack's
complete requirements. The explicit loader downloads the model on first use.

The adapter establishes voices from windows up to two seconds long, sliding at
0.5-second intervals. Windows whose two end probes sound unlike each other do
not seed or train a voice. Repeated overlapping windows are matched without
repeatedly training an already learned identity. It then matches one-second centred probes to the gallery
to label the timeline at 0.5-second resolution, including changes within a
continuous VAD region. This is an offline approximation of speaker boundaries,
not separation of simultaneous voices.

Regions under 0.8 seconds remain `UNKNOWN`; unmatched windows need at least
1.5 seconds to establish a voice. Short earlier regions can match a voice learned
later in the same recording. Unresolved or ambiguous probes remain `UNKNOWN`.
Speaker intervals are represented by exact source sample offsets.

For either `aimara.run_pipeline` or `aimara.run_batch`, append:

```bash
--diarization-backend redimnet2 --diarization-weights /tmp/aimara-redimnet2
```

The default backend is `none`. Use a fresh output directory when switching
diarization on: batch `--resume` does not check the speech configuration.

With diarization enabled, timed words are assigned to speakers by their overlap
with that timeline. By default, whole sentences are labelled with their majority
speaker, or a credible trailing speaker change (at least two words and 0.4 seconds),
matching ROB's sentence grouping policy. Consecutive sentences assigned to the
same speaker are merged. A backend segment without complete word alignment stays
intact; text-only transcripts retain their original VAD region.

For raw word runs with short speaker fluctuations smoothed into the preceding
run, append these options to either runner's command:

```bash
--no-snap-splits-to-sentences \
--min-speaker-run-tokens 2 \
--min-speaker-run-duration 0.4
```

The same policy can be injected into `OfflineROBProcessor`:

```python
from aimara import SpeakerSegmentationConfig, SpeakerSegmenter

segmenter = SpeakerSegmenter(
    SpeakerSegmentationConfig(snap_splits_to_sentences=False)
)
processor = OfflineROBProcessor(
    vad_processor=OfflineVADProcessor(vad, config),
    transcriber=asr,
    diarizer=diarizer,
    speaker_segmenter=segmenter,
)
```

Each group receives its own WAV and corresponding source-video times. Cuts fall
midway between adjacent groups' word boundaries; leading and trailing VAD audio
stay with the outer groups. The WAVs partition the VAD region without duplicated
or discarded samples, even when estimated word times overlap. ASR still runs once
per VAD region, and segment IDs are unique across the whole recording.

Both `aimara.run_pipeline` and `aimara.run_batch` expose the offline segmentation
settings below. Omitting the options keeps Aimara's existing defaults:

| Option | Default | Effect |
| --- | --- | --- |
| `--vad-threshold` | `0.5` | Minimum VAD probability treated as speech. |
| `--min-speech-duration` | `0.2` s | Discard regions with less detected speech. |
| `--min-silence-duration` | `0.5` s | Silence needed to close a speech region. |
| `--pre-buffer-duration` | `0.2` s | Retain source audio before detected onset. |
| `--max-segment-duration` | `30.0` s | Split the detected speech span at this length; the first clip can also include onset pre-buffer. |
| `--snap-splits-to-sentences` / `--no-snap-splits-to-sentences` | enabled | Keep whole sentences or use smoothed word runs. |
| `--min-speaker-run-tokens` | `2` | Minimum word count for a credible speaker change. |
| `--min-speaker-run-duration` | `0.4` s | Minimum span of a credible speaker change. |

Speaker grouping options take effect with diarization enabled. The minimum run
settings smooth short fluctuations in word mode and determine whether a trailing
speaker switch can label a whole sentence in sentence mode. VAD processing stays
mono float32 at 16 kHz with 512-sample windows; timing gates are evaluated at that
window resolution. The effective configurations are logged at startup, and
invalid values, including non-finite durations, fail before model loading.

To select the current ROB ASR configuration's 0.25-second silence gate and
0.5-second onset pre-buffer, append:

```bash
--vad-threshold 0.5 \
--min-silence-duration 0.25 \
--pre-buffer-duration 0.5
```

Use a fresh output directory when changing segmentation settings: batch resume
does not yet check the configuration. `tests/test_segmentation_cli.py` verifies
both runners' defaults, runtime validation, changed VAD boundaries, and sentence
versus word grouping through the offline processor, using model test doubles.

Audio features load each group's WAV. Visual features use the original video
and the same absolute `[start_time, end_time)` interval, for ViT, landmarks, and
expression landmarks. With the current constant-FPS reader, only frames whose
`frame_index / fps` falls in that interval are sampled. Adjacent groups do not
share a boundary frame. An interval with no frames, including one beyond the
video's end, produces a zero visual feature rather than borrowing another
group's frame. Optional active-speaker detection uses the same frame offset and
limits its audio and video to that group, without the previous two-second tail.
The audio encoder retains its existing synthetic silence padding and RMS
normalization.

`tests/test_multimodal_segment_alignment.py` checks this path using a generated
lossless video with numbered frames and known PCM, including A/B/A turns,
fractional boundaries, all three visual modes, and active-speaker offsets.
It uses the real media readers and encoder adapters with model inference replaced
by test doubles; it requires FFmpeg and the unimodal Python dependencies.

## ERC feature bridge

The feature bridge converts the ordered `SpeechSegment` objects into the text,
audio, and visual feature matrices expected by the ERC transformer. The
adapters reuse the existing unimodal model helpers; importing them does not load
any model or download weights.

```python
from aimara import ERCFeatureBridge
from aimara.erc_encoders import (
    OriolAudioEncoder,
    OriolTextEncoder,
    OriolVisualEncoder,
)

bridge = ERCFeatureBridge(
    text_encoder=OriolTextEncoder.load(),
    audio_encoder=OriolAudioEncoder.load(),
    visual_encoder=OriolVisualEncoder.load(),
    speaker_slots=2,
)
features = bridge.build(segments)
model_inputs = features.to_torch_batch(device="cuda")
outputs = model(**model_inputs)
```

Text context is causal and never includes reference sentiment or emotion
labels. Speech segments and diarization intervals trim speaker IDs and normalize
case variants of `unknown` to `UNKNOWN`. Known identities retain their case and
scope (for example, `EUT_speaker1` is not silently merged with `speaker1`). Known
speakers map to stable conversation-local slots in order of first appearance.

The default `unknown_speaker_policy="omit_embedding"` retains UNKNOWN turns and
their text, audio, visual features, and conversation context, while omitting
their speaker embedding. UNKNOWN has a zero speaker-mask row and does not occupy
a known-speaker slot. `to_torch_batch()` supplies a boolean `speaker_known_mask`
when needed; the ERC transformer uses it to zero only the speaker embedding,
leaving the utterance active in attention. This supports two known voices plus
any number of unresolved turns with the existing IEMOCAP checkpoint. It adds no
trained speaker class and does not use the checkpoint's padding row as an
unknown identity. More than two known voices raises a capacity error before
feature extraction, listing the required slots and known speaker IDs.

Both runners expose `--unknown-speaker-policy omit_embedding` (the default).
For earlier runs' behavior, use `--unknown-speaker-policy reserve_slot` or pass
`unknown_speaker_policy="reserve_slot"` to `ERCFeatureBridge`. That policy
reserves slot zero for UNKNOWN and accepts at most one known voice alongside it.
Use a fresh output directory when changing this policy: batch resume does not
check it. Omitting unknown embeddings changes predictions for unresolved turns;
emotion accuracy still needs evaluation. The selected checkpoint must have the
same text, audio, visual, and speaker dimensions as the loaded encoders.

`tests/test_erc_speaker_policy.py` runs the real ERC transformer with a small test
checkpoint. It checks zero unknown embeddings across all nine transformer paths,
retained turn attention and output labels, checkpoint reloading, and unchanged
known-only and legacy calls. These are behavior tests, not an accuracy benchmark.

Run the complete ROB VAD/ASR to IEMOCAP ERC pipeline with:

```bash
LD_PRELOAD=/home/Imatge/media/ssd2/oriol/sdt_env/lib/libpng16.so.16 \
python -m aimara.run_pipeline \
  --video /home/Imatge/media/ssd2/AIMARA/videos/selfie-videos/Anger.mp4 \
  --checkpoint /home/Imatge/media/ssd2/oriol/checkpoints/IEMOCAP/final/vit_softmax.pt \
  --output-dir /tmp/aimara-erc-anger \
  --vad-weights /tmp/aimara-silero \
  --asr-weights /home/Imatge/media/ssd2/AIMARA/models/faster-whisper \
  --asr-model tiny.en \
  --audio-model /home/Imatge/media/ssd2/AIMARA/evaluation_pipeline/models/modelscope/iic/emotion2vec_plus_large \
  --language en \
  --device cpu
```

The `LD_PRELOAD` prefix is currently needed in `aimara_env` because its OpenCV
package otherwise resolves an incompatible bundled `libpng`. It can be removed
once that environment's OpenCV/libpng installation is repaired.

The command writes segment WAV files under `audio/` and the aligned transcript,
timing, emotion, confidence, and class probabilities to `predictions.json`.

## Batch evaluation

The batch runner loads every model once and processes all MP4 files in one
directory. Start the clean selfie-video evaluation in a detachable session:

```bash
tmux new -s aimara-eval

LD_PRELOAD=/home/Imatge/media/ssd2/oriol/sdt_env/lib/libpng16.so.16 \
python -m aimara.run_batch \
  --video-dir /home/Imatge/media/ssd2/AIMARA/videos/selfie-videos \
  --output-dir /home/Imatge/media/ssd2/AIMARA/output/rob-erc-selfie-videos \
  --checkpoint /home/Imatge/media/ssd2/oriol/checkpoints/IEMOCAP/final/vit_softmax.pt \
  --vad-weights /tmp/aimara-silero \
  --asr-weights /home/Imatge/media/ssd2/AIMARA/models/faster-whisper \
  --asr-model tiny.en \
  --audio-model /home/Imatge/media/ssd2/AIMARA/evaluation_pipeline/models/modelscope/iic/emotion2vec_plus_large \
  --language en \
  --device cpu 2>&1 | tee /tmp/aimara-rob-erc-selfie-videos.log
```

Detach with `Ctrl+B`, then `D`, and reconnect with
`tmux attach -t aimara-eval`. Add `--resume` when restarting an interrupted
run. A valid per-video `predictions.json` is reused; incomplete or malformed
results are rerun.

The output root contains combined `predictions.csv`, `predictions.json`, and
`failures.json` files. Each video also receives its own directory containing
segment WAVs and its individual `predictions.json`. A failure is recorded and
does not stop later videos from running.
