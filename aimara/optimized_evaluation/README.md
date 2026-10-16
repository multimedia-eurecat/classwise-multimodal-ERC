# Optimized annotated-input evaluation

This evaluator is maintained in the ERC repository. All executable model and
preprocessing code is Python source in this checkout, including the small
Light-ASD network and sparse 3DDFA geometry helpers with their MIT licenses and
provenance. It needs no sibling repository, private source folder, saved bytecode,
`/tmp` installation or editable package install.

Media, annotations, trained checkpoints and pretrained model assets are separate
inputs. They must be supplied after extracting the repository. Models and video
results are deliberately not committed. Model assets retain their upstream terms.

The inputs contain reference transcripts, speakers and utterance boundaries.
VAD, ASR and diarization are outside this evaluation. The speech pipeline has its
own [quality and benchmark runners](../README.md#reference-quality-and-performance-step-12c).

## Install

The tested main runtime targets Python 3.10 on Linux x86_64. Install system FFmpeg
and FFprobe and create a virtual environment from a working Python 3.10 base.
From the repository root:

```bash
python3.10 -m venv .venv-aimara-evaluation
source .venv-aimara-evaluation/bin/activate
python -m pip install pip==26.0.1 setuptools==80.9.0 wheel==0.45.1
python -m pip install -r aimara/requirements-torch-cpu.txt
python -m pip install -r aimara/requirements-base.txt
```

For CUDA 12.1, substitute `aimara/requirements-torch-cu121.txt` for the CPU Torch
file. For `iemocap-full`, additionally install:

```bash
python -m pip install -r aimara/optimized_evaluation/requirements-landmarks.txt
```

For `meld-full`, use a separate AU environment. Its older py-feat API needs
SciPy 1.11.4, which retains both `scipy.integrate.simps` and
[`scipy.stats.binom_test`](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.stats.binom_test.html) used by py-feat/nltools; do not change the
main environment's SciPy or patch installed libraries:

```bash
python3.10 -m venv .venv-aimara-evaluation-au
.venv-aimara-evaluation-au/bin/python -m pip install \
  pip==26.0.1 setuptools==80.9.0 wheel==0.45.1
.venv-aimara-evaluation-au/bin/python -m pip install \
  -r aimara/requirements-torch-cpu.txt
.venv-aimara-evaluation-au/bin/python -m pip install \
  -r aimara/optimized_evaluation/requirements-au.txt
```

Prepare py-feat's detector weights once while online, before timing a run:

```bash
.venv-aimara-evaluation-au/bin/python -c \
  'from feat import Detector; Detector(au_model="svm", emotion_model="svm", device="cpu")'
```

Py-feat stores these assets in its installed package resources. Preserve them
when moving the runtime to a machine without internet access. Its first detector
construction can otherwise download missing assets and inflate startup timings.

Use the matching CUDA Torch file in that environment for GPU AU inference.
Select it with `--au-python .venv-aimara-evaluation-au/bin/python`.
The main interpreter defaults to the Python running the command; `--sdt-python`
can select another environment. These profiles describe the workstation
evaluation runtime. A phone's platform and inference backend need separate work.

## Supply assets

By default, assets live in `.cache/optimized_evaluation/` and checkpoints in
`checkpoints/`, both relative to the extracted repository. These directories are
ignored by Git. You can keep them elsewhere and pass explicit CLI paths.

```text
.cache/optimized_evaluation/
  huggingface/hub/                       # HF snapshots and refs/main
  modelscope/iic/emotion2vec_plus_large/
    model.pt
    config.yaml
    configuration.json
  light-asd/pretrain_AVA_CVPR.model
  landmarks/                            # only iemocap-full needs these
    mb1_120x120.onnx
    bfm_noneck_v3.pkl
    param_mean_std_62d_120x120.pkl
checkpoints/
  IEMOCAP/final/vit_softmax.pt
  IEMOCAP/final/vit_landmarks_class_reliability_dropout01_va_contrastive.pt
  MELD/final/vit_softmax.pt
  MELD/final/vit_action_units_class_reliability_dropout01_va_contrastive.pt
```

Only the selected checkpoint and feature assets are required. In an online setup
step, populate the two Hugging Face snapshots using the main environment:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

cache = Path('.cache/optimized_evaluation/huggingface/hub')
for model in ('sentence-transformers/all-roberta-large-v1',
              'dima806/facial_emotions_image_detection'):
    snapshot_download(model, cache_dir=str(cache), allow_patterns=[
        '*.json', '*.safetensors', 'pytorch_model.bin', 'vocab*',
        'merges.txt', 'tokenizer*', 'special_tokens_map.json',
    ])
```

Download emotion2vec to its local asset directory using ModelScope:

```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('iic/emotion2vec_plus_large',
                  local_dir='.cache/optimized_evaluation/modelscope/iic/emotion2vec_plus_large')
```

Obtain the AVA pretrained weights from the
[Light-ASD weight directory](https://github.com/Junhua-Liao/Light-ASD/tree/ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight).
For landmarks, supply the MobileNet 120x120 ONNX weights, BFM and 62-parameter
normalization assets from [3DDFA-V2](https://github.com/cleardusk/3DDFA_V2).
The sparse evaluator does not need its dense BFM ONNX decoder, topology file,
FaceBoxes, renderer, native extensions or source checkout.

Alternatively, point `--text-model` and `--vit-model` at complete local HF model
directories. Other overrides are `--assets-root`, `--audio-model`, `--asd-weights`,
`--landmark-model`, `--landmark-bfm`, `--landmark-normalization`, `--checkpoint`
and `--checkpoint-root`. All paths can be relative to your current directory.
The evaluation itself loads local assets; prepare downloads before measuring.

## Run one profile

From the repository root:

```bash
python -m aimara.optimized_evaluation \
  --profile iemocap-base \
  --csv data/input_csv/clean.csv \
  --output-root outputs/optimized_evaluation \
  --device cpu \
  data/videos/clean
```

To invoke a checkout from another directory, use its launcher:

```bash
/path/to/extracted-erc/.venv-aimara-evaluation/bin/python \
  /path/to/extracted-erc/scripts/run_optimized_evaluation.py \
  --profile iemocap-base --csv ./annotations.csv ./videos
```

The shell launcher `scripts/run_optimized_evaluation.sh` uses
`EVAL_LAUNCHER_PYTHON` or `python3`. Caller-relative CLI paths are preserved;
workers import the selected checkout's source even when another checkout exists
in the caller's working directory.

| Profile | Required visual features |
| --- | --- |
| `iemocap-base` | ViT |
| `iemocap-full` | ViT + landmarks |
| `meld-base` | ViT |
| `meld-full` | ViT + action units |

Every profile also uses text and emotion2vec audio. A single checkpoint runs.
Use repeated `--video Anger.mp4` flags for a diagnostic subset, `--dry-run` to
inspect the stage graph without loading models or writing outputs, and `--help`
for the complete CLI. With `--use-predefined-csv`, the annotation default is
`<video_dir>/../../input_csv/<video-folder-name>.csv`; `--input-csv-dir` overrides
that lookup. `--run-name` changes the output name without changing annotation lookup.

Outputs are `<output-root>/<run-name-or-video-folder>/<profile>/`, containing
`configuration.json`, `timing_report.json`, `pseudo_gt.csv`, embeddings, stage
logs, a selected-profile pickle and `results_<dataset>_<model>.json`.
Old workstation outputs remain independent and can be retained as historical runs.

## Timing, correctness and comparison

`pipeline_rtf` is the sum of all stage process wall times divided by the actual
unique video durations from FFprobe. It includes imports, model startup,
preprocessing/cache construction, inference and stage writes. Preflight
validation/checksums and final report writing are outside this number.
Synchronized phase times separate startup and processing; visual preprocessing
contains its nested decode, face detection and active-speaker phase times.
Parent and child phase times must not be added together. Model residency and
streaming optimization are subsequent steps.

Memory records parent peak RSS and Torch allocator peaks per stage. FFmpeg child
RSS and non-Torch GPU allocations are excluded. `--device auto|cpu|cuda` selects
the device. The resolved device, visible GPUs, runtime package versions and
native-library/thread settings are recorded and fingerprinted.

AU inference uses py-feat's path API and the same selected-face cache as ViT.
The audio span passed to active-speaker detection matches its decoded video
window. `--asd-lookahead` retains the configurable two-second default: this is
future context that contributes to prediction delay. Missing detections are
counted, all-zero streams and non-finite/missing features fail the evaluation.

Resume requires `--resume`, matching source/asset/input/configuration identities
and valid artifact hashes. Changed settings need fresh outputs. Rerunning a stage
invalidates its successors. Failed and resumed attempts have no benchmark RTF.

For continuity with the original evaluation, checkpoint architecture, label maps
(including IEMOCAP `joy` to `happy`), whole-dialogue scoring exclusions, and the
legacy dataset reader's speaker encoding are retained. In particular, that reader
interprets `M` as its first speaker and other strings as its second speaker; numeric
annotation speaker IDs are not normalized here. This is a known legacy scoring
limitation, separate from the portable evaluator's performance measurements.
The newer speech quality evaluator has different turn and label filtering rules.

## Test

```bash
python -m unittest discover -s tests -p 'test_optimized_evaluation.py' -v
```

Tests include relocation outside the original checkout, no bytecode/private-path
dependencies, the four profile graphs, asset diagnostics, AU inputs, alignment,
resume and RTF validity. Real-checkpoint validation is recorded separately; small
synthetic smoke inputs do not establish mobile latency or production accuracy.

Validation on 2026-10-16: 284 repository tests passed, including 26 evaluator
checks. Both full profiles ran from a relocated copy with private-code imports
blocked; all text/audio/ViT/landmark/AU embeddings matched the previous outputs.
Four checkpoint parity checks reproduced 24 saved predictions and their rounded
probabilities exactly. A new isolated CPU AU environment passed `pip check`,
imports and real 20-value AU inference using cached binary assets, with no copied
package code or library patches. See [validation_receipt.json](validation_receipt.json).
