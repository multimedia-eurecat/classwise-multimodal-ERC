# AIMARA integration

## Actual ROB engine replay → ERC

Use `run_rob_native` when the purpose is to rehearse consumption of **ROB's
actual segmentation**, rather than evaluate our separately adapted offline
segmenter. It loads the unchanged `VADEngine`, `RediVoiceEngine`,
`VoiceIdentityManager`, and `ASREngine`/`ParakeetASREngine` from a separate
`EutSpeechAudioProcessing` checkout. The default required revision is
`fd89b8b537c17b7ccd103310039ddcf6faf7aa4e`. A different HEAD or modified tracked
speech/config files is rejected; source/config SHA-256 hashes are recorded.
`--rob-revision` is an explicit opt-in to another reviewed revision, not a
guarantee that its interface is compatible.

ROB owns audio buffering, silence timers, speaker-change flushes, provisional
and final identity observations, ASR word timing, UNKNOWN grace, and sentence
grouping. The speech-activity `_vad_callback` and `_on_eut_speaker_changed`
method bodies are loaded unchanged from ROB's `DiarizationNode` source, without
importing ROS. Only message delivery and publisher plumbing are replaced with
direct callbacks. This path does **not** call our `OfflineVADProcessor`,
`ROBSpeakerDiarizer`, `SpeakerSegmenter`, or adapted transcriber wrappers.

The checked-in ROB YAML supplies the settings: currently Whisper `turbo`,
batched inference, the `en, es` language allow-list, 0.25 s ASR silence timeout,
and 0.5 s onset pre-buffer. Overrides are explicit and saved. ROB selects its
own device and applies its own precision fallback; for example, GTX 1080 Ti
falls back from requested float16 to float32. `--device` controls ERC only;
use `CUDA_VISIBLE_DEVICES` to choose the GPU visible to ROB.

From the repository root, using the existing Whisper or CUDA runtime profile:

```bash
python -m aimara.run_rob_native \
  --rob-repo /path/to/EutSpeechAudioProcessing \
  --video /path/to/selfie-videos/*.mp4 \
  --weights /tmp/aimara-rob-native-models \
  --checkpoint /path/to/vit_softmax.pt \
  --output-dir /tmp/aimara-native-results
```

Use `--audio-model /path/to/emotion2vec_plus_large` when the audio encoder is
already cached locally. `--asr-backend parakeet` selects ROB's actual optional
Parakeet engine and requires the Parakeet runtime dependencies. `--speech-only`
omits ERC and does not require a checkpoint. Models download into the writable
weights/cache directories on first use; supplying an existing cached ROB
weights directory avoids downloading them again. ROS and DIART are unnecessary
for this REDI native replay.

Each recording gets a fresh worker process and a fresh, **session-only** voice
gallery. Shared MongoDB identities, cross-robot synchronization, and robot
self-voice filtering are outside this recording test, even if enabled in the
deployment YAML. Those differences are explicit in `rob_capture.json`.
VAD runs in its own spawned process, matching the separation of ROB's VAD node;
otherwise GPU work in the other engines can artificially delay audio delivery.
An explicit 512-sample silence warmup and a 2.016 s silence pre-roll precede
the file with default settings; at least 2 s of tail silence permits final
turns and timers to finish. Speech is fed at 512 samples per 32 ms, using real
clocks and actual background workers. Delivery lateness and maximum VAD backlog
are recorded. Queue overflow, logged engine errors and unfinished workers fail
the replay instead of producing an apparently successful observation.

The read-only group observer exports the **final** grouping attempt after
ROB's speaker-grace retries, and forwards ROB's existing transcript callback
arguments unchanged. It exports internal input-start wall clocks plus group
offsets. Publication timestamps alone cannot provide these boundaries.
`observe_groups(engine, callback)` in `rob_native_export.py` can also be attached
to a live ROS-owned engine before audio starts; it does not replace that node's
normal transcript publication. A ROS integration must additionally establish
the original media clock for alignment. The hook observes engine groups before
ROS-node self-voice filtering; if robot speech is active, a deployment collector
must also honor that node's accepted/dropped transcript decisions.

ERC retains ROB's text, groups, identities, gaps and overlaps. It extracts WAVs
and visual features over the exported source intervals, rounds to audio samples,
and clips only to physical media edges. `erc_bounds.json` records those changes.
ROB's internal clocks/word estimates are not human-verified word boundaries.
The exported groups are ordered by start time for the ERC conversation, without
merging adjacent groups or smoothing their labels. Literal UNKNOWN is normalized
for ERC's existing omit-embedding policy.

The checkpoint still supports **two known speakers plus UNKNOWN**. If ROB emits
three known identities, that video's speech results are preserved and its ERC
failure is recorded; identities are never collapsed to force a prediction.
Other videos continue. The command exits nonzero when any ERC video fails, even
when all speech replays succeeded. ERC processes each completed conversation;
this is not yet incremental online emotion inference.

Outputs are:

- `rob_native.csv`: the same six columns as the reference CSV; emotions are blank
  for speech-only runs or explicitly failed ERC videos.
- `run_report.json`: overall result, per-video group counts, timing and ERC status.
- `index.html`: portable inspection with original videos, four aligned timeline
  lanes (reference turns, ASR input regions, raw speaker activity, final ERC
  groups), clickable intervals, extracted audio, transcripts, emotions, and
  per-video failures. Overlapping intervals are stacked; playback moves the
  timeline cursors together.
- `<video>/rob_capture.json`: pinned source fingerprints, selected settings,
  raw final groups, speech-activity events and VAD observations.
- `<video>/rob.log`, `audio/*.wav`, `erc_bounds.json`, `predictions.json`, and
  `erc_result.json`: diagnostics, exact extracted samples, and ERC results.

`--import-only` consumes successful saved captures for the selected videos and
reruns only materialization/ERC. It rejects ASR overrides because it cannot
change how those captures were produced. There is no implicit native replay
resume: an ordinary invocation performs new speech runs. Keep a separate output
directory when comparing configurations.

This provides actual-engine integration evidence. **ROS message scheduling,
QoS, deployment identity galleries, target robot hardware, and live ERC latency
still require an on-stack integration test.**

The 15 October 2026 validation used the configured Whisper `turbo`, language
allow-list and batching on the CUDA runtime (effective float32 on GTX 1080 Ti).
All **10 speech replays and 10 ERC conversations passed**, producing **33 final
groups**. Maximum audio-delivery lateness was 3.45 ms; no audio chunk was more
than 32 ms late and no VAD queue overflowed. The acceptance suite passed
**227 tests with zero skips**. The versioned
[`rob_native_results.json`](rob_native_results.json) preserves source hashes,
settings, versions, per-video outputs, and artifact fingerprints. These are
integration observations, not an ERC accuracy or live-latency benchmark.

On this workstation, the portable inspection page and six-column CSV are under
`docs/segmentation_review/rob_native/whisper_final/`. With the existing viewer
server and SSH tunnel, open
`http://localhost:8000/rob_native/whisper_final/`. The CSV is `rob_native.csv`.
The main segmentation viewer also links to this native-engine run. Earlier
development attempts are preserved separately and are not the validated run.

Supply `--reference-csv /path/to/selfie-videos.csv` to `run_rob_native` to include
annotations in the review. This affects visualization only. To rebuild an
existing report without replaying speech or rerunning ERC:

```bash
python -m aimara.rob_native_review \
  --output-dir docs/segmentation_review/rob_native/whisper_final \
  --reference-csv /path/to/selfie-videos.csv
```

The CSV is copied to `reference.csv`, and `review_timelines.json` records its
hash, the rendering source hash, input capture/bounds hashes, and plotted
intervals. Subsequent rebuilds can use the saved snapshot without the flag or
access to the original video directory. The current native review uses the
**30-turn** reference; the older main viewer retains its original 31-turn
evaluation snapshot.

**Raw speaker timeline means diarization activity before final sentence
grouping**, not emotion predictions or ASR segments. For these saved native
captures, it is a retrospective reconstruction from ROB's activity events in
delivery order. Heartbeats extend active runs, stop events close them, and
backdated labels revise earlier intervals. It is not the exact internal ASR
speaker timeline at every inference instant: those intermediate states and
event arrival times were not captured. Uncolored raw gaps mean no recorded
active identity; they do not by themselves establish silence or missing speech.
Reference IDs 0/1 are independent of ROB's session speaker IDs. The ASR lane
deduplicates recorded input bounds shared by final groups; empty/unpublished
transcription calls are absent from these older captures.

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

## Isolated runtime environments

The runtime profiles target **Python 3.10 on Linux x86_64**. Create one virtual
environment for Whisper and another for Parakeet; each contains the complete ERC
pipeline because the runners load speech and emotion models in one process.
Run installation commands from the repository root. Install the system `ffmpeg`
package (including `ffprobe`) first; Parakeet's source dependencies also need a
C/C++ compiler and Python development headers.

If `python3.10` is not on `PATH`, use the full path to a compatible interpreter.
On this workstation, `/home/Imatge/media/ssd2/oriol/au_env/bin/python` supplies a
working Python 3.10.20 base for the Parakeet venv. The new venv uses its own
packages. A venv still inherits the base interpreter's native standard-library
modules: the `aimara_env` Python's ICU-enabled SQLite conflicts with the system
C++ library after CPU Torch loads. This appears as `CXXABI_1.3.15 not found` in
the NeMo/IPython import path despite `pip check` passing. Use a compatible base
Python rather than upgrading packages to try to solve that native-library issue.

For the CPU Whisper environment:

```bash
python3.10 -m venv .venv-aimara-whisper
source .venv-aimara-whisper/bin/activate
python -m pip --isolated install --index-url https://pypi.org/simple \
  pip==26.0.1 setuptools==80.9.0 wheel==0.45.1
python -m pip --isolated install -r aimara/requirements-torch-cpu.txt
python -m pip --isolated install --index-url https://pypi.org/simple \
  -r aimara/requirements-whisper.txt
python -m aimara.check_environment --asr-backend whisper --device cpu
python -m unittest discover -s tests -q
```

For Parakeet, create `.venv-aimara-parakeet` instead and replace the final runtime
requirements file with `aimara/requirements-parakeet.txt`. Check it with
`python -m aimara.check_environment --asr-backend parakeet --device cpu`.
The Parakeet profile includes Whisper so either backend is selectable in that
environment. The Whisper profile does not install NeMo. Avoid `--system-site-packages`
and installing these profiles into the existing shared `aimara_env`.

`constraints-runtime.txt` pins the resolved direct and transitive dependencies
for both profiles. Each requirements file applies it automatically; constraints
for optional dependencies do not install those packages in the Whisper profile.
Keep the build-tool pins above when recreating an environment. To update the
profiles, resolve their union in a fresh Python 3.10 environment with the chosen
Torch build, refresh the constraints (preserving flavor-independent Torch pins),
and repeat the checks and model smoke runs. These are version pins rather than
artifact hashes; source-built dependencies also depend on the host compiler.

For CUDA 12.1, install `aimara/requirements-torch-cu121.txt` in place of the CPU
Torch file, then install the selected runtime profile. Check with `--device cuda`.
The three Torch packages must use matching build flavors. The installation
follows the [official Torch version matrix](https://pytorch.org/get-started/previous-versions/).
Faster-Whisper also needs cuBLAS and cuDNN visible to the system loader; follow
its [GPU library instructions](https://github.com/SYSTRAN/faster-whisper#gpu).
The supplied CUDA Torch build uses cuDNN 9. Jetson/ARM and other Python versions
need their own dependency resolution and validation.

The profiles retain Torch 2.4.1 and use `facenet-pytorch==2.5.3`, whose declared
dependencies accept that Torch/Torchvision pair. They share Transformers 4.51.3,
within NeMo 2.4's declared range, and NumPy 1.26.4. Only
`opencv-python-headless==4.10.0.84` should provide `cv2`; installing multiple
OpenCV distributions into one environment mixes their files. These profiles
cover the default text/emotion2vec/ViT path and ReDimNet2; research-only AU,
landmark, training, and ROS packages are outside their scope.

`check_environment` reports package versions, `pip check` results, system
executables, runtime import failures, and small NumPy/Torch/OpenCV/face-detector
operations. It loads the face detector's bundled weights but does not download
speech or ERC checkpoints. CUDA checks verify device visibility; actual GPU
inference still needs a model run. Reuse model caches through the existing CLI
weight arguments and `HF_HOME`/`MODELSCOPE_CACHE` variables; keep caches separate
from virtual environments. Existing predictions keep their schema and checkpoints
keep their architecture. Dependency changes are not part of the resume
fingerprint, so use a new output directory for comparisons across environments.

Validation on 2026-10-14: both CPU profiles installed in isolated `/tmp` venvs,
with clean `pip check` results and 184 passing tests. Runtime probes exercised
the real libraries, including NeMo ASR and speaker-model imports in the Parakeet
profile. The fresh Whisper profile ran the cached `Anger.mp4` through VAD,
transcription, all three encoders, and the real ERC checkpoint without
`LD_PRELOAD`. Compared with the shared environment, both turns' transcripts,
timings and emotion labels matched; the largest probability difference was
about `1.34e-7`. The existing shared environment was not changed. Step 12B below
records the subsequent pretrained CPU/CUDA validation, including Parakeet and
AmberNet.

## ROB speech dependencies

The VAD adapter uses PyTorch and the Silero model loaded through Torch Hub. The
default Whisper ASR adapter uses the same Faster-Whisper version pinned by the ROB repository:

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

Both runners expose the Whisper inference controls below. Existing runs keep
single-region decoding and the previous CPU fallback unless options are changed:

| Option | Default | Effect |
| --- | --- | --- |
| `--asr-compute-type` | `float16` | Requested CTranslate2 model precision. |
| `--asr-cpu-compute-type` | `int8` | CPU fallback when `float16` or `int8_float16` is requested; `float32` selects ROB's CPU fallback policy. |
| `--asr-batched-inference` / `--no-asr-batched-inference` | disabled | Enable Faster-Whisper's batched decoder. |
| `--asr-batch-size` | `16` | Maximum chunks decoded together within one VAD region when batching is enabled. |
| `--asr-beam-size` | `5` | Beam search size for either decoder. |

To select ROB's batched inference and CPU fallback settings, append:

```bash
--asr-compute-type float16 \
--asr-cpu-compute-type float32 \
--asr-batched-inference \
--asr-batch-size 16
```

The loader also accepts `cpu_compute_type`, `use_batched_inference`, `batch_size`,
and `beam_size` as keyword arguments. It logs the selected device, effective
compute-type argument, and decoding settings; the returned transcriber exposes
`inference_config`, `device`, and `effective_compute_type`. Unsupported requested
precision still gets the existing float32 retry. Invalid options fail before
model loading.

Both decoders keep word timestamps enabled and internal VAD disabled. Batched
decoding supplies contiguous input-relative clips of at most 30 seconds,
covering the full VAD region, including onset pre-buffer and longer configured
regions. This avoids Faster-Whisper 1.2.1 rejecting an input of at least 30 seconds
or truncating an oversized explicit clip. Alignment state resets between regions
and videos, including after decoder failures. Models remain loaded across batch
videos; the ASR batch size controls chunks within each region.

Faster-Whisper 1.2.1's batched decoder uses different prompt and temperature
fallback behavior from its standard decoder. Changing batching, precision, or
beam size can change transcripts and word boundaries; compare emotion accuracy
as well as runtime. Batch `--resume` reruns results when these settings change.
`tests/test_rob_asr.py` and
`tests/test_whisper_cli.py` cover options, CPU fallback, chunk coverage, timing
retention, state reset after lazy failures, and both runner paths with test doubles.

### Optional Parakeet transcriber

Both runners accept `--asr-backend whisper` (the default) or
`--asr-backend parakeet`. The optional backend uses NeMo and the default
[`nvidia/parakeet-tdt-0.6b-v3` model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
It supports English and Spanish among 25 European languages; Catalan requires
Whisper. Explicit unsupported language requests fail before any model loads.

Install the complete optional runtime in the isolated Parakeet environment
described above, after installing the matching Torch packages:

```bash
python -m pip --isolated install --index-url https://pypi.org/simple \
  -r aimara/requirements-parakeet.txt
```

That file pins `nemo_toolkit[asr]==2.4.0`, matching ROB, together with the ERC
runtime and compatible optional dependencies.
Importing `aimara.rob` and running Whisper do not import or require NeMo.

To select Parakeet, append these options to an existing runner command:

```bash
--asr-backend parakeet \
--parakeet-model nvidia/parakeet-tdt-0.6b-v3 \
--language en,es
```

`--asr-weights` is the backend's cache directory. Parakeet loads the cached
Hugging Face `.nemo` checkpoint or downloads it on explicit loading. A local file
can be selected with `--parakeet-model /path/to/model.nemo`. Use
`--parakeet-local-files-only` to require cached or local checkpoints. The model
runs in float32 on the selected CPU/CUDA device and warms up once; disable warm-up
with `--no-parakeet-warmup`. The Whisper model, precision, beam, and batching
options apply only to the Whisper backend.

Parakeet automatically transcribes the input language and does not accept a
language-forcing argument. Here `--language en` supplies an assumed output label;
it does not establish that English was detected. For `auto` or a comma-separated
list, the optional `langid_ambernet` identifier selects the label from supported
candidates. Configure it with `--parakeet-language-id-model` (a NeMo model name
or local `.nemo` file) and `--parakeet-language-id-min-confidence` (default `0.9`).
An empty identifier name disables it. For local-only operation, the identifier
also needs a cached or local checkpoint. Unavailable identification, low
confidence, or audio shorter than 0.25 seconds leaves `Transcript.language`
unset. No previous recording's label is reused. A language label cannot certify
that unsupported-language speech was transcribed correctly.

```python
from aimara.rob import ParakeetConfig, ParakeetTranscriber

asr = ParakeetTranscriber.from_nemo(
    weights_dir="/path/to/parakeet-weights",
    config=ParakeetConfig(language="en,es"),
    device="cuda",
)
transcript = asr.transcribe(segment_audio, sample_rate=16000)
```

The adapter requires mono float32 audio at 16 kHz, as supplied by the offline
processor. It converts NeMo frame offsets using the encoder subsampling factor
and preprocessor stride, or retains already converted seconds. Word text receives
leading spaces so the shared sentence/word segmenter can use it. Each word is
assigned to at most one backend sentence; incomplete sentence metadata falls
back to a whole transcript span while preserving the text. Missing alignment
retains the full input interval. A NeMo word-confidence aggregation failure is
retried once with confidence disabled, then the decoding settings are restored.

`tests/test_parakeet.py` and `tests/test_parakeet_cli.py` cover these adapters,
optional loading, backend selection, and speaker-aware WAV/prediction output
with model test doubles. Step 12B also exercised trained Parakeet and AmberNet
on CPU and CUDA with the real feature encoders and ERC checkpoint. Step 12C below
reports comparative runtime, transcript accuracy, and emotion accuracy. Batch
`--resume` reruns results when the selected backend or its settings change.

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

The default backend is `none`. Batch `--resume` reruns results when switching
diarization on or off.

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

Batch `--resume` reruns results when active segmentation settings change.
`tests/test_segmentation_cli.py` verifies
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
Batch `--resume` reruns results when this policy changes.
Omitting unknown embeddings changes predictions for unresolved turns;
step 12C reports a small paired comparison of these policies. The selected
checkpoint must have the same text, audio, visual, and speaker dimensions as
the loaded encoders.

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
when using the validated isolated CPU profiles above, which import OpenCV
without that workaround.

The command writes segment WAV files under `audio/` and the aligned transcript,
timing, emotion, confidence, and class probabilities to `predictions.json`.

### Speech metadata in prediction output

Each new prediction includes the fields below in per-video JSON, combined JSON,
and batch CSV. The existing `confidence` field remains the predicted emotion
class probability; speech scores are separate diagnostics and do not filter
turns or become emotion-model inputs.

| Field | Meaning |
| --- | --- |
| `asr_backend`, `asr_model` | Backend and configured model identifier (`whisper` plus its model alias, or `parakeet` plus its repository/local checkpoint). Unknown for legacy/custom transcripts without metadata. |
| `asr_device`, `asr_compute_type` | Selected device and effective model-loading precision argument, when known. |
| `transcript_confidence` | Backend-reported score for the whole ASR input. Whisper reports `null`; Parakeet reports the mean of valid NeMo word-confidence values when available. Zero is a valid score. |
| `transcript_confidence_method` | `mean_word_confidence` when a Parakeet score is present; otherwise `null`. |
| `speaker_id_confidence` | Selected identity's match scores weighted by their scored coverage of the emitted segment. `-1` means unavailable; zero is a valid reported score. |
| `language_source` | `configured`, `detected`, `model`, or `fallback`; `null` when unavailable. A configured Parakeet label is an assumption, while `model` denotes Whisper's English-only model restriction. |
| `asr_input_start_time`, `asr_input_end_time` | Source-video boundaries of the original VAD interval sent to ASR. Every speaker group split from that decode shares these references and its transcript confidence. |
| `start_time`, `end_time` | Source-video boundaries of this group's own WAV and visual interval. |
| `confidence` | Emotion class probability, with its existing meaning. |

Unavailable optional values are JSON `null` and blank CSV cells. The speaker
score retains its explicit `-1` sentinel. Parakeet confidence errors trigger the
existing retry without confidence; that recovered transcript has no score.
Missing, malformed, or non-finite word scores also leave confidence unavailable
while preserving text. These ASR and identity scores are not calibrated emotion
probabilities and should not be compared directly with `confidence`.

Speaker confidence uses `sum(overlap_samples * match_score) / segment_samples`
for scored spans carrying the selected known identity. Gaps, other voices, and
unscored spans reduce scored coverage; entirely unscored or UNKNOWN segments
report `-1`. ReDimNet2's combined speaker intervals retain their individual
probe confidence spans, so a score change or unscored stretch is not lost when
adjacent labels are merged.

For custom transcribers, attach immutable metadata explicitly:

```python
from aimara import ASRMetadata, Transcript

transcript = Transcript(
    "hello",
    language="en",
    metadata=ASRMetadata(
        backend="custom",
        model="model-id",
        transcript_confidence=None,
        language_source="configured",
    ),
)
```

`Transcript(text, language)` remains valid and defaults to unavailable metadata.
The offline processor carries `Transcript.metadata` into each segment's
`asr_metadata` and computes speaker confidence from the source-aligned timeline.
`tests/test_speech_metadata.py` covers score availability, evidence coverage,
ASR-input references, and JSON/CSV export through the pipeline. Adapter tests
also cover language provenance and Parakeet's reported confidence and retry.
Older cached predictions can lack these fields. Batch `--resume` reruns legacy
results without a compatible resume manifest or prediction schema.

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
run. Each successful video receives a `resume_manifest.json` alongside its
`predictions.json`. Results are reused only when the manifest version, pipeline
configuration, source video identity, prediction checksum, and prediction schema
match. Missing, legacy, incompatible, or damaged caches are rerun automatically.
Valid empty predictions (videos without detected speech) can also be reused.

The configuration covers the active ASR backend/model/language and inference
settings, effective ASR device/precision, VAD settings, enabled diarization and
speaker grouping, UNKNOWN policy, ERC profile/checkpoint/device, and loaded
text/audio/visual encoder configurations and available model revision IDs.
Output locations, `--resume`, and options for inactive backends do not affect it.
Changing settings in the same output directory replaces that video's results;
use separate output directories to retain comparison runs.

Explicit checkpoint files, including local `.nemo` files, use SHA-256 content
identities. Model cache directories use file names, sizes, and modification times
to avoid reading every large weight file; lock files, download temporaries, and
Python bytecode are ignored. Source videos use their resolved path, size, and
modification time. These checks do not detect edits to videos or directory-cached
weights that preserve both size and modification time, or an upstream model
changing under the same ID without a recorded revision or local cache change.
Identities are captured once after model loading; keep inputs and weights stable
during a batch. Runtime library/code changes are not automatically fingerprinted.

Manifests are written atomically only after successful predictions. A failed
rerun removes its old manifest so a later resume retries that video, while other
successful videos remain reusable. `tests/test_batch.py` and
`tests/test_batch_resume_config.py` cover reuse, configuration changes, checkpoint
replacement, input changes, legacy/corrupt caches, empty outputs, and recovery.

For a custom Python pipeline, supply `AIMARAPipeline(..., resume_config=...)` or
`run_batch(..., run_config=...)` with all settings that affect its predictions.
Use `Path` values for model artifacts to include their identities. Without a
configuration, batch processing still works but resume reruns cached videos.

## Acceptance validation: step 12A

Activate an isolated runtime environment from the setup section, then run from
the repository root:

```bash
python -m aimara.run_acceptance --report /tmp/aimara-acceptance-report.json
```

Add `--verbose` to print every test name. The command runs the full existing
regression suite plus the generated-media acceptance cases. FFmpeg and FFprobe
are required. Exit codes are `0` for a complete pass, `1` for a failed or
incomplete suite (including skips/expected failures), and `2` for missing media
executables. An empty suite also fails. Ordinary `unittest` discovery remains
available, but the acceptance runner enforces the stricter completion gate.

The report records the fixture-manifest SHA-256, Python/platform/package versions,
individual test names, statuses, timings and failure details, along with totals.
Failed subtests mark their parent test as failed. Reports are written atomically;
fixtures, generated media, and tiny checkpoint files are created in temporary
directories and cleaned up after the run. Intentional failure/recovery tests
produce error logs; the final acceptance status and report identify their outcome.

The fixed [fixture manifest](../tests/fixtures/acceptance_cases.json) defines:

| Case | Required behavior |
|---|---|
| Silence | Empty predictions; ASR and feature extraction are skipped |
| Speech at source boundaries | Speech at time zero and the partial final audio chunk survives without added source samples |
| Speech longer than 30 seconds | VAD splits conserve every source speech sample |
| Speaker A → B → A | Turn order, speaker identity and aligned clips survive through ERC and JSON |
| Two known speakers plus UNKNOWN | All turns reach ERC; UNKNOWN has a zero speaker-mask row |
| Missing faces | Speech remains; the visual feature is zero with full-frame fallback disabled |
| Excess known speakers | A two-speaker checkpoint rejects three known identities without writing emotion results |
| Batch recovery and resume | Invalid media/capacity failures are recorded, later videos run, repaired inputs recover, matching caches are reused, and changed configurations rerun |

These cases use lossless PCM markers and numbered video frames. Controlled
doubles supply VAD probabilities, ASR words, speaker attribution and unimodal
neural embeddings. The real FFmpeg decoder, VAD state machine, speaker grouping,
WAV writer, text/audio/visual adapters, feature bridge, checkpoint loader, ERC
transformer, pipeline and batch writer execute. Assertions check source PCM,
source-frame windows, ASR-input intervals, chronological context, finite normalized
six-class probabilities and matching JSON/CSV exports. The tiny ERC checkpoint
is seeded and untrained; its output labels serve contract validation.

Step 12B below covers pretrained-model smoke runs, including ReDimNet2 and
Parakeet/AmberNet weights and available CUDA hardware. Step 12C covers reference
accuracy, configuration comparisons, latency and memory measurements. Passing
12A establishes the deterministic integration contracts; those model-quality
and deployment checks remain separate validation stages.

## Pretrained-model smoke validation: step 12B

`run_model_smoke` loads the actual pretrained models and runs the normal batch
pipeline with observations of ASR words, language-ID probabilities and speaker
intervals. It accepts the same runtime options as the normal runners:

```bash
python -m aimara.run_model_smoke \
  --video-dir /home/Imatge/media/ssd2/AIMARA/videos/selfie-videos \
  --reference-csv /home/Imatge/media/ssd2/AIMARA/input_csv/selfie-videos.csv \
  --output-dir /tmp/aimara-model-smoke-whisper \
  --checkpoint /home/Imatge/media/ssd2/oriol/checkpoints/IEMOCAP/final/vit_softmax.pt \
  --vad-weights /tmp/aimara-silero \
  --asr-weights /home/Imatge/media/ssd2/AIMARA/models/faster-whisper \
  --audio-model /home/Imatge/media/ssd2/AIMARA/evaluation_pipeline/models/modelscope/iic/emotion2vec_plus_large \
  --device cpu
```

Select a subset by repeating `--video`, for example `--video Anger.mp4 --video
Happy.mp4 --video Neutral.mp4`. Use a fresh output directory for each configuration.
The runner stages symlinks for a selected subset; original media and annotations
remain in their source directories. Weights load once and stay available across
videos. Normal JSON/CSV predictions, failures, and resume manifests are retained.

Append `--diarization-backend redimnet2 --diarization-weights
/tmp/aimara-12b-models/redimnet2` to exercise the pretrained speaker model.
For Parakeet/AmberNet, replace `--asr-weights` with
`/tmp/aimara-12b-models/parakeet` and append:

```bash
--asr-backend parakeet --language auto --require-language-id
```

The explicit loaders fetch missing weights on first use. After populating the
cache, add `--parakeet-local-files-only` to require local Parakeet and AmberNet
checkpoints. `--require-language-id` fails if the identifier cannot load or no
successful probability-producing identifier calls occur. An output language
can still be unset when the score falls below the configured threshold; it does
not reuse a previous input's label. For CUDA, activate the pinned CUDA profile,
expose its cuBLAS/cuDNN libraries as described in the setup section, select an
available GPU with `CUDA_VISIBLE_DEVICES`, and use `--device cuda`.

Each configuration produces `smoke_report.json`; each video also receives
`model_observations.json`. Reports preserve source/configuration identities,
model-artifact SHA-256 hashes, package versions, actual precision/device,
reference-annotation checksum, raw ASR words, language-ID probabilities, speaker
timelines and prediction checks. Timings are observations from a smoke run;
controlled latency/memory comparisons belong to step 12C.

Smoke gates check chronological word timestamps bounded by the ASR input (with
a 0.2-second tolerance for decoder frame rounding), exact segment WAV content
and source bounds, sequential segment IDs, finite normalized six-class ERC
probabilities, and nonempty predictions for annotated speech. The first
successful video is repeated after the batch to verify that turn/text/language/
speaker/emotion outputs remain stable; probabilities may differ by at most
`1e-5`. Use `--no-check-state-reset` only for a run that intentionally omits that
check. Model-load failures and per-video failures are recorded, and the command
exits nonzero when the selected configuration fails its gates.

Real runs exposed Silero recurrence leaking between recordings: `Happy.mp4`
produced different VAD regions after different preceding videos. The adapter now
resets Silero at each offline recording's start while retaining recurrence
between chunks within that recording. Regression tests cover both boundaries,
and the real clip was checked in both orders. Resume cache version 2 invalidates
older manifests because this correction can change segmentation and predictions.

The versioned [validation receipt](model_smoke_results.json) records the final
CPU/CUDA matrix on the clean English selfie-video set, with the default
`tiny.en` Whisper and `nvidia/parakeet-tdt-0.6b-v3` Parakeet model. CUDA runs used
GTX 1080 Ti cards with the pinned CUDA 12.1 profile. Whisper's requested float16
precision fell back to float32 on these cards; reports record that effective
precision. Parakeet used float32. The underlying speech decisions and all three
feature encoders executed with pretrained weights; no model doubles were used.

| Final configuration | Videos | Emotion segments | Smoke result |
|---|---:|---:|---|
| Whisper CPU | 10 | 20 | Passed |
| Whisper + ReDimNet2 CPU | 10 | 26 | Passed |
| Whisper CUDA | 10 | 20 | Passed |
| Parakeet + AmberNet CPU | 3 | 6 | Passed |
| Parakeet + AmberNet + ReDimNet2 CUDA | 10 | 27 | Passed |

All 43 video/configuration runs and all five repeated-input state checks passed
after the Silero correction. Both CPU environments and the CUDA environment
also passed 202 acceptance tests with zero skips. Final detailed artifacts are
under `/tmp/aimara-12b-v2/<configuration>/`; the compact receipt is versioned in
the repository. Original model weights and input media remain outside it.

Reference speaker diagnostics compare dominant timeline labels per annotated
turn and recognize A → B → A up to an identity-label permutation. These quality
diagnostics are reported independently from smoke gates. Short middle turns
can remain UNKNOWN, and some turns can share an identity. The final corrected
runs recovered the annotated A → B → A pattern in four of ten clips in both
diarization configurations. Step 12C below quantifies those attribution errors
along with transcript/emotion accuracy and runtime. This set covers
English clips and the tested hardware; other languages, noise conditions and
hardware still need their own validation.

## Reference quality and performance: step 12C

`evaluate_quality` scores completed pretrained smoke reports against the existing
`selfie-videos.csv` annotations. It loads no models and changes neither annotations
nor predictions. Source-video and annotation checksums must agree across reports;
failed, incomplete, duplicate or mismatched input reports are rejected. Every
pair of configurations is compared on its shared videos, so a three-video CPU
run or four-video policy comparison cannot silently compete with a ten-video run.

```bash
python -m aimara.evaluate_quality \
  --reference-csv /home/Imatge/media/ssd2/AIMARA/input_csv/selfie-videos.csv \
  --smoke-report /tmp/aimara-12c/whisper-redimnet2-cuda/smoke_report.json \
  --smoke-report /tmp/aimara-12b-v2/parakeet-redimnet2-cuda/smoke_report.json \
  --report /tmp/aimara-quality-report.json
```

Methods and scoring scope:

- ASR: micro word error rate over chronological raw ASR calls. Text is Unicode
  normalized and casefolded; punctuation is removed and internal apostrophes
  retained. Exported transcripts are scored separately to expose words lost
  during speaker grouping. Reference utterance boundaries supply no word-level
  alignment, so this evaluation makes no word-timestamp accuracy claim.
- Speakers: optimal one-to-one identity mapping per video, with separate correct,
  uncovered, UNKNOWN and confused seconds. Both raw diarization and exported
  segments are scored. The annotations span whole turns including pauses; these
  duration scores are not standard speech-masked DER. Change points use
  one-to-one matching within 0.5 seconds, including changes to/from UNKNOWN.
- Emotion: duration-weighted mean segment probabilities within each reference
  turn, followed by argmax. Entirely uncovered turns count as incorrect, and
  partial coverage is recorded per turn. Accuracy and confusion counts use
  supported reference labels; macro F1 averages their six present classes.
  Scoring uses annotations only after inference, with no ground-truth segments
  or transcripts fed to the model.

The default emotion map accepts the six checkpoint labels plus `anger→angry`
and `sadness→sad`. It excludes `joy`, `fear`, `disgust` and `surprise`: the
reference taxonomy has no exact checkpoint equivalent for them. Of the 31
annotated turns, 20 are scorable; the 11 excluded turns are 3 joy, 3 fear,
2 disgust and 3 surprise. An explicit `--emotion-map /path/to/map.json` replaces
this policy; all mapping targets must be valid checkpoint labels. The model
still emits its usual six labels for excluded turns.

Measured reference results on all ten clips (default settings unless stated):

| Configuration | Raw ASR WER | Exported speaker duration correct | Emotion accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Whisper CPU, diarization disabled | 2.30% | — | 60% | 54.52% |
| Whisper + ReDimNet2 CPU | 2.30% | 68.50% | 75% | 69.44% |
| Whisper CUDA, diarization disabled | 1.84% | — | 60% | 54.52% |
| Whisper + ReDimNet2 CUDA | 1.84% | 68.64% | 70% | 62.30% |
| Parakeet + AmberNet + ReDimNet2 CUDA | 0.92% | 71.29% | 60% | 51.98% |
| Whisper + ReDimNet2 CUDA, raw word splits | 1.84% | 70.47% | 55% | 49.15% |
| Whisper + ReDimNet2 CUDA, minimum run 1 token / 0 seconds | 1.84% | 66.71% | 65% | 59.42% |
| Whisper + ReDimNet2 CUDA, batched ASR | 1.84% | 68.65% | 70% | 62.30% |

Exported transcript WER equals raw ASR WER in these runs. CUDA Whisper and
Parakeet have identical raw speaker timelines: 69.60% correct duration,
10.472 uncovered seconds, 7.140 UNKNOWN seconds, and 6.708 confused seconds
over 80 annotated seconds. Their word/sentence grouping produces different
exported speaker scores. Whisper recovers 13 of 21 exported speaker changes
within 0.5 seconds; Parakeet recovers 15. Boundary precision is 92.86% and 93.75%
respectively. Pauses contribute to uncovered time because the reference spans
are turn annotations, not a speech activity mask.

The reserved-UNKNOWN-slot comparison uses only Anger, Disgust, Joy and Surprise,
whose baseline outputs require at most one known speaker plus UNKNOWN. Across
their five scorable emotion turns, `omit_embedding` gets 3/5 correct and
`reserve_slot` gets 4/5. This is a small diagnostic: reserving a slot still
cannot represent two known voices plus UNKNOWN. The selected `omit_embedding`
default remains appropriate for that capacity requirement; this comparison
does not establish an accuracy advantage for it.

These observations support retaining sentence snapping and the existing
minimum-run settings for now. Parakeet's lower transcript WER does not translate
into higher ERC accuracy on this set. CPU/CUDA feature precision also changes
some emotion decisions; comparisons must retain their recorded environment.
No runtime defaults or model weights were changed by this validation.

For controlled timings, `benchmark_pipeline` rebuilds one passed configuration
in a fresh process. Use a fresh output directory, local model caches and the
same isolated environment as the source report. Run configurations sequentially
with fixed thread counts; CUDA runs additionally need the driver/library setup
described above. CPU measurements use the CPU-only profiles; encoders normally
choose devices from Torch availability, so a CPU benchmark rejects an accidental
mix with CUDA encoders rather than describing it as a CPU-only measurement.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m aimara.benchmark_pipeline \
  --smoke-report /tmp/aimara-12c/whisper-redimnet2-cuda/smoke_report.json \
  --video Anger.mp4 --video Neutral.mp4 --repeat 2 \
  --output-dir /tmp/aimara-benchmark-whisper
```

The Linux benchmark records model construction time, one excluded warmup per
clip, then sequential round-robin repeats with CUDA synchronization. Timed
inference includes audio/video decoding, neural models and WAV/JSON writes;
validation and checksum hashing run outside the timer. Outputs must remain
stable against warmup within the smoke probability tolerance. The report saves
individual samples and computes real-time factor as total processing seconds
divided by total audio seconds. Model construction includes lazy imports and
loader-internal warmups, with warm disk caches; it excludes initial CLI imports
and first-use model downloads.

Memory measurements cover startup and inference: parent-process peak RSS
(`ru_maxrss`), sampled RSS, sampled driver-level GPU memory including
CTranslate2, and separate Torch allocator peaks. GPU sampling waits 250 ms
between queries and can miss brief spikes. FFmpeg child-process RSS is excluded.
Memory sampling errors fail the benchmark instead of reporting missing memory
as zero. Each run records package versions, model hashes, source identity,
configuration, CPU affinity, thread settings and GPU properties.

Sequential measurements used Anger and Neutral (eight seconds each), one warmup
per clip and two measured repeats per clip, four CPU threads, an i7-5820K host,
and the same physical GTX 1080 Ti for every CUDA configuration:

| Configuration | Model construction | Warm median per video | Real-time factor | Peak parent RSS | Sampled GPU peak |
|---|---:|---:|---:|---:|---:|
| Whisper CPU | 8.96 s | 16.18 s | 2.019 | 5.19 GiB | — |
| Parakeet + AmberNet CPU | 35.38 s | 16.51 s | 2.063 | 7.85 GiB | — |
| Whisper CUDA, diarization disabled | 10.84 s | 7.18 s | 0.897 | 5.17 GiB | 2.50 GiB |
| Whisper + ReDimNet2 CUDA | 11.18 s | 9.57 s | 1.200 | 5.17 GiB | 2.59 GiB |
| Whisper + ReDimNet2 CUDA, batched ASR | 11.26 s | 9.44 s | 1.181 | 5.17 GiB | 2.65 GiB |
| Parakeet + AmberNet + ReDimNet2 CUDA | 37.96 s | 9.70 s | 1.207 | 6.26 GiB | 5.25 GiB |

Real-time factor below 1 means total processing was faster than the input audio
duration in this offline benchmark; it is not a streaming latency measurement.
The small batched timing difference is insufficient to establish a throughput
advantage. The host was shared; runs were sequential for this validation but
hardware was not exclusively reserved.

All six benchmarks passed, including stable output checks for 24 measured
samples after 12 warmup calls. Reference scoring covers 87 video/configuration
observations, including 44 new ablation runs. The final acceptance suite passed
210 tests with zero skips in each of the CPU Whisper, CPU Parakeet and CUDA
profiles. Metric tests cover known word-error counts, identity permutation,
UNKNOWN/missing attribution, competing boundary matches, duration-weighted
emotion scoring, excluded labels and provenance failures.

The versioned [quality/performance receipt](quality_results.json) preserves
compact metrics and detailed-report checksums. Detailed outputs live under
`/tmp/aimara-12c/`, with earlier baseline observations under `/tmp/aimara-12b-v2/`.
The checks establish completed measurements, not a production accuracy gate.
This ten-clip English synthetic set cannot establish general accuracy, language
ID calibration, robustness to noise, or deployment latency on other hardware.
Its eight-second clips also do not establish batching throughput on long speech.
A separate held-out target-domain set and agreed quality thresholds are needed
before selecting production defaults.

## Segmentation CSV comparison and diagnostic replay

`compare_segmentation` exports saved speaker-aware outputs in the reference CSV's
`video_file,start,end,speaker,text,emotion` format and traces which stage retains
or loses each reference text-turn change. It does not load neural models or
change runtime segmentation settings:

```bash
python -m aimara.compare_segmentation \
  --reference-csv /media/ssd2/AIMARA/input_csv/selfie-videos.csv \
  --original-reference-csv /media/ssd2/AIMARA/input_csv/selfie-videos_orig.csv \
  --smoke-report /tmp/aimara-12c/whisper-redimnet2-cuda/smoke_report.json \
  --smoke-report /tmp/aimara-12b-v2/parakeet-redimnet2-cuda/smoke_report.json \
  --output-dir /tmp/aimara-segmentation-comparison
```

Outputs include `new_pipeline_whisper.csv`, `new_pipeline_parakeet.csv`,
`comparison_by_reference_turn.csv`, `boundary_diagnosis.csv`, `word_trace.csv`,
`segmentation_stages.csv`, `diagnostic_replays.csv`, `summary.csv`, a provenance
record in `diagnosis.json`, and a browser-readable `index.html`. The `emotion`
column in the two pipeline CSVs is the saved model prediction, not the reference
label. Diagnostic replays contain no recomputed emotion labels. Reference-turn,
reference-boundary and output-group identifiers are zero-based.

The actual grouping is replayed first and must exactly reproduce saved group
times, text and speaker labels. Controlled alternatives reuse the same ASR
observations: sentence majority without the credible-trailing-speaker override,
unsmoothed word runs, and an explicitly annotation-based reference-speaker
timeline. That last replay is a counterfactual, not an independent model result.
Reference text is aligned to saved ASR words by global edit alignment; English
number words zero through ten and their digits are treated as equivalent for
this structural comparison. This avoids shifting a turn boundary because one
transcript says `four. All right` and another says `4. Alright`. It does not
change the WER definition or results reported in step 12C.

The current CSV has 30 turns; the 14 October evaluation used the preserved
31-turn snapshot. The change merges/relabels the early `Surprise.mp4` turns.
Both snapshots are retained by the comparison; historical validation receipts
remain unchanged. These annotations include pauses and have no independently
verified word times. Uncovered reference duration and small endpoint differences
therefore require audio review and do not establish missing speech.

On the saved CUDA runs, both backends retain multiple ASR text segments in
`Excited.mp4` and `Happy.mp4`, but the raw speaker assignments cause the final
speaker groups to collapse. For `Sadness.mp4`, the Whisper replay loses a turn
change through the credible-trailing-speaker rule; disabling only that override
restores it with identical saved words and diarization. This local replay
isolates a grouping effect, while the acoustic correctness of the competing
word and diarization times still requires manual inspection. Six diagnostic
tests cover this distinction, exact replay, numeric alignment, pause handling
and rejection of ambiguous ASR-input associations.

For normal pipeline batch runs, the output root contains combined
`predictions.csv`, `predictions.json`, and
`failures.json` files. Each video also receives its own directory containing
segment WAVs and its individual `predictions.json`. A failure is recorded and
does not stop later videos from running.

## Transcript-aligned emotion evaluation

`evaluate_emotions.py` evaluates ground-truth turns **1 and 3** for each video.
It joins exact video basenames and aligns all three reference transcripts to
the predicted text in order. Reference timings and speaker IDs do not influence
matching. The phone response anchors alignment but is excluded from scoring.
Split, merged, and cross-boundary predictions produce one result row per
target-turn/predicted-segment overlap, including the matched text fragments.

Using the existing AIMARA runtime from the repository root:

```bash
python -m aimara.evaluate_emotions \
  --reference /media/ssd2/AIMARA/input_csv/selfie-videos.csv \
  --prediction IEMOCAP=docs/emotion_evaluation/model_runs/iemocap/predictions.csv \
  --labels IEMOCAP=happy,sad,neutral,angry,excited,frustrated \
  --prediction MELD=docs/emotion_evaluation/model_runs/meld/predictions.csv \
  --labels MELD=neutral,surprise,fear,sadness,joy,disgust,anger \
  --output-dir docs/emotion_evaluation
```

The shared, configurable `emotion_circumplex.json` contains engineering
prototype angles, not empirical class centroids. Angular score is
`1 - shortest_angle_degrees / 180`. Neutral has no angle: angular fields remain
blank, while the overall score gives neutral-neutral 1 and neutral-emotional
mismatches 0. Joy and happy are label synonyms; excited remains distinct.
Each target's score weights its matched segments by matched-token count;
overall means give each eligible target equal weight. Text coverage is reported
separately, with an optional coverage-adjusted score. Entirely unmatched targets
remain in the denominator with score 0. Categorical accuracy measures
matched-token-weighted canonical-label correctness within each target.

`summary.csv` includes each checkpoint's supported ground-truth subset and
their common canonical-label subset. `matched_segments.csv` preserves every
overlap, raw/canonical labels, eligibility, angular error, score, and model
provenance. `target_utterances.csv`, `by_emotion.csv`, `unused_predictions.csv`,
and `evaluation.json` provide aggregation, exclusions, methodology, and hashes.
The output also snapshots the exact reference and angle mapping.

`run_emotion_checkpoints.py` runs the local IEMOCAP and MELD checkpoints on the
saved native ROB segments, sharing text/audio/visual features and preserving
all speech inputs. It does not rerun speech inference. Its CLI requires
`--input-dir`, `--model-cache`, and a fresh `--output-dir`; use `--help` for
device/checkpoint options. Existing inference outputs are never overwritten.
On this workstation the established runtime is
`/media/ssd2/oriol/sdt_env/bin/python`. The supplied CSV is evaluated separately
under `docs/emotion_evaluation/supplied_baseline/` because its original checkpoint
identity was not recorded. `docs/emotion_evaluation/reproduce.sh` re-scores the
saved predictions without rerunning either checkpoint.

## Optimized annotated-input evaluator

The maintained implementation now lives in [optimized_evaluation/](optimized_evaluation/README.md).
Use `python -m aimara.optimized_evaluation` from the repository root. The guide
covers fresh environments, required weights, all four profiles and portable
asset/runtime options; the evaluator no longer needs the workstation evaluation
folder, Python bytecode, or private model-source checkouts. This is the
reference-annotated evaluation; the ROB speech pipeline remains separate.
