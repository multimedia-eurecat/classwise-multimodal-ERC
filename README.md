# Multimodal Emotion Recognition in Conversations via Class-Wise Adaptive Modality Fusion and Affective Geometry

Code for the paper *"Multimodal Emotion Recognition in Conversations via Class-Wise Adaptive Modality Fusion and Affective Geometry"* (ECCV 2026 submission, paper ID #24). The method extends the Self-Distillation Transformer (SDT) [Ma et al., 2024] for multimodal Emotion Recognition in Conversations (ERC) on IEMOCAP and MELD.

> **Branch note:** `feature/aimara-speech-integration` extends the original ERC
> repository with an offline integration of ROB's VAD and ASR components. The
> goal is to prepare the ERC model for eventual use inside ROB's processing
> pipeline. The original training and evaluation code remains available and is
> not replaced by this integration.

## AIMARA speech integration

The original ERC workflow expects utterance boundaries, transcripts, speaker
IDs, and extracted multimodal features to exist before inference. This branch
adds the missing runtime path from an unsegmented video to emotion predictions:

```text
video
  -> mono 16 kHz audio
  -> ROB Silero VAD
  -> speech regions
  -> ROB Faster-Whisper ASR
  -> SpeechSegment objects
  -> text, audio, and visual feature encoders
  -> ERC transformer checkpoint
  -> timestamped emotion predictions
```

Only the Python logic required for VAD and ASR was adapted from
[`Eurecat/eut_speech_audio_processing`](https://github.com/Eurecat/eut_speech_audio_processing).
ROS is deliberately kept outside this branch. A future ROB adapter can translate
between ROS messages and the Python contracts without coupling the emotion model
to ROS during development.

### What this branch adds

| Area | Original repository | This branch |
|---|---|---|
| Input | Pre-segmented dataset utterances | Complete video files or finalized speech segments |
| Speech boundaries | Supplied by MELD/IEMOCAP metadata | ROB-compatible Silero VAD |
| Transcripts | Supplied by dataset annotations | ROB-compatible Faster-Whisper ASR |
| Runtime contract | Dataset-specific pickle structures | `SpeechSegment` and `ConversationFeatures` |
| Feature extraction | Dataset preparation scripts | Runtime text, emotion2vec, and visual adapters |
| Inference | Dataset dataloaders | Direct checkpoint inference over ROB segments |
| Execution | Research scripts | Single-video and resumable batch CLIs |

The integration lives under [`aimara/`](aimara/). Its main components are:

- `rob/vad.py` and `rob/asr.py`: ROS-free ports of the required ROB behavior.
- `offline_processor.py`: audio decoding, VAD segmentation, ASR, and segment WAV materialization.
- `erc_features.py` and `erc_encoders.py`: alignment and multimodal feature extraction.
- `erc_model.py`: checkpoint validation and timestamp-aligned emotion predictions.
- `pipeline.py`: complete ROB-to-ERC orchestration.
- `run_pipeline.py` and `run_batch.py`: single-video and resumable directory execution.

Detailed setup, commands, model locations, output formats, and the current
OpenCV environment workaround are documented in
[`aimara/README.md`](aimara/README.md).

### Current validation status

- The complete pipeline has been exercised on the synthetic `Anger.mp4` video.
- A batch run processed all 10 clean `selfie-videos` without pipeline failures.
- Per-video and combined JSON/CSV outputs preserve VAD timing, transcripts,
  emotion predictions, confidence values, and class probabilities.
- The automated suite currently contains 54 passing tests covering contracts,
  VAD/ASR adapters, segmentation, feature alignment, checkpoint validation,
  orchestration, failure handling, and resume behavior.

The synthetic run is an integration stability check, not evidence that ROB
improves ERC accuracy. The selected IEMOCAP checkpoint exposes six classes
(`happy`, `sad`, `neutral`, `angry`, `excited`, and `frustrated`), so it cannot
directly predict synthetic labels such as `fear`, `disgust`, or `surprise`.

### Known limitations

- Diarization is not integrated; runtime segments currently use speaker ID
  `UNKNOWN`, mapped consistently to speaker slot zero.
- Processing is offline and video-oriented. ROB integration will require a thin
  in-memory or ROS-facing adapter around finalized speech segments.
- VAD regions can contain more than one conversational turn when pauses are
  short, especially without diarization.
- The deployed emotion taxonomy and checkpoint must be selected as part of the
  ROB interface contract.
- The current `aimara_env` OpenCV package requires a temporary `libpng`
  `LD_PRELOAD` workaround described in the AIMARA README.

### Next integration steps

1. Define the stable ROB-facing request and emotion-result schemas.
2. Add an in-memory inference entry point that avoids unnecessary intermediate files.
3. Select and document the production checkpoint and emotion taxonomy.
4. Measure initialization cost, per-segment latency, and memory use on target hardware.
5. Add the final ROS adapter after the Python boundary is stable.

## Abstract
 
ERC requires integrating heterogeneous textual, audio, and visual cues while accounting for conversational context and emotional dynamics. We extend SDT with appearance+geometry visual representations, class-wise adaptive modality fusion, and a valence-arousal prior for affective transitions. On MELD and IEMOCAP, geometry-enhanced visual representations improve weighted F1 by 0.27 and 4.36 points over appearance-only features, respectively, while class-wise adaptive fusion provides further gains of 0.17 and 0.25 points over the original softmax gate. The valence-arousal prior yields targeted improvements of 0.30 and 0.74 accuracy points on emotionally shifted utterances while preserving performance on stable turns.

<p align="center">
  <img src="assets/emotion_shifts_erc.png" width="800" alt="Method overview">
</p>

## Contributions
 
1. **Appearance+geometry visual representations** — ViT appearance features combined with facial geometry descriptors (3D landmarks, expression parameters, or action units) via symmetric addition, strengthening the visual stream.
2. **Class-wise adaptive modality fusion** — replaces SDT's softmax-based multimodal gate with a fusion strategy that estimates each modality's reliability separately per emotion class, operating at the logit level.
3. **Valence-arousal prior for emotion shifts** — a post-hoc correction grounded in Russell's circumplex model, applied to the fused logits after class-wise fusion. It scales with the magnitude of the predicted affective transition (`β_i`) and favors classes close to the previous affective state (`τ_va`), controlled by a global strength parameter `α`.


## Results (weighted F1, mean over MELD + IEMOCAP)
 
<div align="center">

| Configuration | MELD | IEMOCAP | Mean |
|---|---|---|---|
| Updated SDT baseline | 75.49 | 69.50 | 72.50 |
| + Appearance+geometry visual stream | 75.76 | 73.86 | 74.81 |
| + Class-wise adaptive modality fusion | **75.93** | **74.11** | **75.02** |

</div>
 
The valence-arousal prior is evaluated separately on emotion-shift subsets (see paper Table 4): it improves shift-utterance accuracy by 0.30 points on MELD and 0.74 points on IEMOCAP, while leaving stable-utterance accuracy essentially unchanged.
 
 ## Repository structure
 
```
.
├── multimodal/           # SDT-based model: dataloader, model, training, inference
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
├── unimodal/              # Per-modality embedding extraction
│   ├── text/              # RoBERTa-large (sentence-transformers, all-roberta-large-v1)
│   ├── audio/              # emotion2vec_plus_large + degradation protocol
│   └── visual/             # ViT + 3D landmarks / expression params / action units + active speaker detection
├── aimara/                 # ROB VAD/ASR to ERC runtime integration
├── tests/                  # AIMARA integration unit and orchestration tests
├── LICENSE
└── README.md
```

See [`multimodal/README.md`](multimodal/README.md),
[`unimodal/README.md`](unimodal/README.md), and
[`aimara/README.md`](aimara/README.md) for details on each stage.

## Datasets
 
Experiments use **MELD** (7-class, multi-party, 13,708 utterances, official train/val/test splits) and **IEMOCAP** (6-class, dyadic, 7,433 utterances, sessions 1-4 train / session 5 test). Neither dataset is redistributed here — obtain them from their original sources under their respective licenses and set the paths expected by `unimodal/*/main.py` and `multimodal/dataloader.py`. IEMOCAP's continuous valence-arousal annotations are used to derive dataset-specific class centroids for the valence-arousal prior; MELD uses canonical Russell circumplex coordinates instead, as it has no continuous annotations.

## Pipeline
 
1. **Extract unimodal features** for text, audio, and visual streams — see [`unimodal/README.md`](unimodal/README.md).
2. **Train / evaluate the multimodal model** on top of the extracted features — see [`multimodal/README.md`](multimodal/README.md).
