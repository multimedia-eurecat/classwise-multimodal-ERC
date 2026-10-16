"""Run the selected repository ERC model on merged annotated-input features."""

import sys
from .eval_metrics import phase
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from multimodal.model import Transformer_Based_Model, MaskedNLLLoss, MaskedKLDivLoss
from multimodal.dataloader import IEMOCAPDataset, IEMOCAP_VA_MAP

# ── emotion label names ───────────────────────────────────────────────────────
IEMOCAP_EMOTIONS = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
MELD_EMOTIONS    = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]

# ── visual encoder → D_visual ─────────────────────────────────────────────────
_VISUAL_CONFIGS = {
    "vit":               768,
    "landmarks":         408,
    "landmarks_exp":     39,
    "action_units":      60,
    "vit_landmarks":     1176,
    "vit_landmarks_exp": 807,
    "vit_action_units":  828,
}


def parse_checkpoint_name(path: str) -> dict:
    """
    Try to infer model hyperparameters from the checkpoint filename.
    Expected pattern: {encoder}_{fusion}_dp{dropout}_ca{alpha}.pt
    Returns a dict of inferred values (may be empty if pattern doesn't match).
    """
    import re
    name = os.path.splitext(os.path.basename(path))[0]
    hints = {}

    # visual encoder
    for enc in sorted(_VISUAL_CONFIGS.keys(), key=len, reverse=True):
        if enc in name:
            hints["visual_encoder"] = enc
            break

    # fusion mode
    for fm in ["scalar_reliability", "class_reliability", "transformer_reliability",
               "entropy", "softmax"]:
        if fm in name:
            hints["fusion_mode"] = fm
            break

    # dropout
    m = re.search(r"dp([\d.]+)", name)
    if m:
        hints["modal_dropout"] = float(m.group(1))

    # circumplex alpha
    m = re.search(r"ca([\d.]+)", name)
    if m:
        hints["circumplex_alpha"] = float(m.group(1))

    return hints


def build_model(args, n_classes: int, n_speakers: int, device: torch.device):
    """Construct Transformer_Based_Model from CLI args."""
    D_visual = _VISUAL_CONFIGS[args.visual_encoder]

    _LANDMARK_ENCODERS = {"landmarks", "landmarks_exp", "action_units"}
    use_landmark_mlp = bool(args.landmark_mlp) and args.visual_encoder in _LANDMARK_ENCODERS

    from multimodal.dataloader import IEMOCAP_VA_MAP
    va_map = None
    if args.circumplex_alpha > 0 and args.dataset == "IEMOCAP":
        va_map = IEMOCAP_VA_MAP

    model = Transformer_Based_Model(
        args.dataset,
        args.temp,
        args.D_text,
        D_visual,
        args.D_audio,
        args.n_head,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        n_speakers=n_speakers,
        dropout=args.dropout,
        modal_dropout=0.0,                  # no dropout at inference
        dropout_strategy=args.dropout_strategy,
        fusion_mode=args.fusion_mode,
        con_temperature=0.0,                # disable contrastive at inference
        use_disentangle=False,
        use_intra_visual_gate=bool(args.intra_visual_gate),
        use_intra_visual_gate_learned=bool(args.intra_visual_gate_learned),
        use_speaker_embeddings=bool(args.speaker_embeddings),
        use_landmark_mlp=use_landmark_mlp,
        gate_dim=args.gate_dim,
        circumplex_alpha=args.circumplex_alpha,
        va_map=va_map,
        circumplex_tau=args.circumplex_tau,
    )
    return model, D_visual


def main():
    ap = argparse.ArgumentParser(
        description="Run a trained SDT model on inference videos."
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to .pt checkpoint file")
    ap.add_argument("--dataset",    default="IEMOCAP", choices=["IEMOCAP", "MELD"])
    ap.add_argument("--pkl",        required=True,
                    help="Path to generated_erc.pkl")
    ap.add_argument("--out",        required=True,
                    help="Path to save JSON results")

    # ── model architecture (must match the checkpoint) ────────────────────────
    ap.add_argument("--visual-encoder",             default="vit_landmarks")
    ap.add_argument("--fusion-mode",                default="softmax",
                    choices=["softmax", "entropy", "scalar_reliability",
                             "class_reliability", "transformer_reliability"])
    ap.add_argument("--intra-visual-gate",          type=int, default=0, choices=[0, 1])
    ap.add_argument("--intra-visual-gate-learned",  type=int, default=0, choices=[0, 1])
    ap.add_argument("--landmark-mlp",               type=int, default=0, choices=[0, 1])
    ap.add_argument("--speaker-embeddings",         type=int, default=1, choices=[0, 1])
    ap.add_argument("--dropout-strategy",           default="input")
    ap.add_argument("--circumplex-alpha",           type=float, default=0.0)
    ap.add_argument("--circumplex-tau",             type=float, default=0.3)
    ap.add_argument("--gate-dim",                   type=int, default=64)
    ap.add_argument("--D_text",     type=int, default=1024)
    ap.add_argument("--D_audio",    type=int, default=1024)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--n_head",     type=int, default=8)
    ap.add_argument("--dropout",    type=float, default=0.3)
    ap.add_argument("--temp",       type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-cuda",    action="store_true", default=False)

    args = ap.parse_args()

    # Try to infer architecture hints from checkpoint filename
    hints = parse_checkpoint_name(args.checkpoint)
    if hints:
        print("Inferred from checkpoint name:", hints)
        for k, v in hints.items():
            attr = k.replace("-", "_")
            if not hasattr(args, attr) or getattr(args, attr) in (0, 0.0, "softmax", "vit_landmarks"):
                setattr(args, attr, v)

    requested = os.environ.get("EVAL_DEVICE", "auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cuda = torch.cuda.is_available() and not args.no_cuda and requested != "cpu"
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Device: {'GPU' if cuda else 'CPU'}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}  visual_encoder: {args.visual_encoder}  "
          f"fusion: {args.fusion_mode}")

    n_classes  = 7 if args.dataset == "MELD" else 6
    n_speakers = 9 if args.dataset == "MELD" else 2
    emotion_names = MELD_EMOTIONS if args.dataset == "MELD" else IEMOCAP_EMOTIONS

    # Build model
    with phase('model_load'):
        model, D_visual = build_model(args, n_classes, n_speakers, device)
        print(f"  D_visual={D_visual}  params={sum(p.numel() for p in model.parameters()):,}")

        # Load checkpoint
        state = torch.load(args.checkpoint, map_location=device)
        # Older full checkpoints include a fixed VA lookup even when the
        # inference graph does not use the optional circumplex loss.
        if "va_coords" in state and "va_coords" not in model.state_dict():
            state = {k:v for k,v in state.items() if k != "va_coords"}
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing:
            print(f"  [WARN] Missing keys in checkpoint: {missing[:5]}")
        if unexpected:
            print(f"  [WARN] Unexpected keys in checkpoint: {unexpected[:5]}")
        model.to(device).eval()
        print("Checkpoint loaded.")

    testset = IEMOCAPDataset(
        args.pkl,
        train=False,
        va_json=None,
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        collate_fn=testset.collate_fn,
        num_workers=0,
        shuffle=False,
    )
    print(f"Test set: {len(testset)} dialogues")

    loss_function = MaskedNLLLoss()
    kl_loss       = MaskedKLDivLoss()

    # ── Run inference ─────────────────────────────────────────────────────────
    results_by_dialogue = {}

    with phase("erc_inference"), torch.no_grad():
        for data in test_loader:
            if len(data) == 8:
                textf, visuf, acouf, qmask, umask, label, shift_severity, vid_names = [
                    d.to(device) if isinstance(d, torch.Tensor) else d for d in data
                ]
            else:
                textf, visuf, acouf, qmask, umask, label = [
                    d.to(device) for d in data[:-1]
                ]
                shift_severity = None
                vid_names = data[-1]

            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1
                       for j in range(len(umask))]

            _, _, _, _, all_prob, _, _, _, _, _, _ = model(
                textf, visuf, acouf, umask, qmask, lengths,
                labels=None,
                shift_severity=shift_severity,
            )

            probs = all_prob                              # (B, N, C)
            preds = torch.argmax(probs, dim=-1)          # (B, N)

            for b, vid_name in enumerate(vid_names):
                n_valid = lengths[b]
                pred_ids = preds[b, :n_valid].cpu().tolist()
                true_ids = label[b, :n_valid].cpu().tolist()
                prob_mat = probs[b, :n_valid].cpu().tolist()

                utterances = []
                for i in range(n_valid):
                    utterances.append({
                        "utterance_id":   i,
                        "predicted":      emotion_names[pred_ids[i]],
                        "ground_truth":   emotion_names[true_ids[i]],
                        "correct":        pred_ids[i] == true_ids[i],
                        "probabilities":  {
                            emotion_names[c]: round(float(prob_mat[i][c]), 4)
                            for c in range(n_classes)
                        },
                    })

                n_correct = sum(u["correct"] for u in utterances)
                results_by_dialogue[vid_name] = {
                    "utterances": utterances,
                    "accuracy":   round(n_correct / n_valid * 100, 1),
                }

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Results  |  checkpoint: {os.path.basename(args.checkpoint)}")
    print("=" * 60)

    all_correct = 0
    all_total   = 0

    for vid_name in sorted(results_by_dialogue.keys()):
        r = results_by_dialogue[vid_name]
        utts = r["utterances"]
        print(f"\n[{vid_name}]  accuracy={r['accuracy']:.1f}%")
        for u in utts:
            tick = "✓" if u["correct"] else "✗"
            top_pred = u["predicted"]
            gt       = u["ground_truth"]
            conf     = u["probabilities"][top_pred]
            print(f"  utt {u['utterance_id']}  {tick}  "
                  f"pred={top_pred:<12} gt={gt:<12}  conf={conf:.3f}")
        all_correct += sum(u["correct"] for u in utts)
        all_total   += len(utts)

    overall_acc = round(all_correct / all_total * 100, 1) if all_total else 0.0
    print(f"\n{'='*60}")
    print(f"  Overall accuracy: {all_correct}/{all_total} = {overall_acc}%")
    print(f"{'='*60}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "checkpoint":    args.checkpoint,
        "dataset":       args.dataset,
        "visual_encoder": args.visual_encoder,
        "fusion_mode":   args.fusion_mode,
        "overall_accuracy": overall_acc,
        "n_correct":     all_correct,
        "n_total":       all_total,
        "dialogues":     results_by_dialogue,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
