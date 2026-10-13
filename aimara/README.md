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
The offline processor currently uses the joined text; splitting VAD regions using
word timings and the speaker timeline is the next integration step.

The models are composed by the offline processor, which writes one WAV and
returns one `SpeechSegment` per VAD region:

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

This first adapter labels existing VAD regions using up to their latest two
seconds of audio. It does not split a region when the speaker changes inside
it. Regions under 0.8 seconds are left `UNKNOWN`; unmatched regions need at
least 1.5 seconds to establish a voice. An earlier short region can match a
voice established later in the same recording, but unresolved or ambiguous
regions remain `UNKNOWN`. Speaker intervals use exact source sample offsets.

For either `aimara.run_pipeline` or `aimara.run_batch`, append:

```bash
--diarization-backend redimnet2 --diarization-weights /tmp/aimara-redimnet2
```

The default backend is `none`. Use a fresh output directory when switching
diarization on: batch `--resume` does not check the speech configuration.

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
labels. Known speaker IDs map to stable conversation-local slots; `UNKNOWN`
maps to slot zero and reserves that slot. The current two-speaker checkpoint
therefore accepts two known voices, or one known voice plus `UNKNOWN`; more
distinct labels raise an explicit error. Handling unknown turns alongside two
known voices and splitting regions at speaker changes are subsequent integration
steps. The selected checkpoint must have the same text, audio, visual, and
speaker dimensions as the loaded encoders.

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
