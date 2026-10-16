"""Merge only the selected profile features into the legacy ERC pickle layout."""

import os
import pickle

import numpy as np
import pandas as pd

RUN_ROOT = os.environ["EVAL_RUN_ROOT"]
CSV_PATH = os.environ["EVAL_CSV_PATH"]
EMB_DIR = os.environ["EVAL_EMB_DIR"]

# Input PKLs
TEXT_PKL   = os.path.join(EMB_DIR, "text.pkl")
AUDIO_PKL  = os.path.join(EMB_DIR, "audio.pkl")
VISUAL_VIT_PKL = os.path.join(EMB_DIR, "visual_vit.pkl")
VISUAL_LMK_PKL = os.path.join(EMB_DIR, "visual_landmarks.pkl")
VISUAL_AU_PKL  = os.path.join(EMB_DIR, "visual_action_units.pkl")


# IEMOCAP label mapping (6 classes: happy=0, sad=1, neutral=2, angry=3, excited=4, frustrated=5)
_EMOTION_MAP = {
    "happy":      0,
    "joy":        0,   # joy → happy (nearest IEMOCAP class)
    "sad":        1,
    "sadness":    1,
    "neutral":    2,
    "angry":      3,
    "anger":      3,
    "excited":    4,
    "frustrated": 5,
    "fear":       None,   # no IEMOCAP equivalent → fallback
    "disgust":    None,   # no IEMOCAP equivalent → fallback
    "surprise":   None,   # no IEMOCAP equivalent → fallback
}

# For emotions with no IEMOCAP equivalent, assign neutral=2
_FALLBACK_LABEL = 2


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _dim(emb: dict) -> int:
    d = next(iter(emb))
    u = next(iter(emb[d]))
    return emb[d][u].shape[0]


def build_pkl(
    df: pd.DataFrame,
    video_files: list,
    video_to_did: dict,
    text_emb: dict,
    audio_emb: dict,
    visual_emb1: dict,
    visual_emb2: dict | None,
    dim_text: int,
    dim_audio: int,
    dim_vis1: int,
    dim_vis2: int,
    output_path: str,
    label_map: dict,
    fallback_label: int,
    label_name: str,
):
    """Build and save one SDT-compatible PKL with visual = emb1 + emb2."""
    dim_visual = dim_vis1 if visual_emb2 is None else dim_vis1 + dim_vis2

    videoIDs      = {}
    videoSpeakers = {}
    videoLabels   = {}
    videoText     = {}
    videoAudio    = {}
    videoVisual   = {}
    videoSentence = {}
    testVid       = []
    unmapped      = set()

    for vfile in video_files:
        did  = video_to_did[vfile]
        name = os.path.splitext(vfile)[0]

        group    = df[df["video_file"] == vfile].sort_values("utterance_id")
        T        = len(group)
        utt_ids  = group["utterance_id"].tolist()
        speakers = [str(s) for s in group["speaker"].tolist()]
        texts    = group["text"].astype(str).tolist()

        labels = []
        for emo in group["emotion"].astype(str).str.strip().str.lower():
            label = label_map.get(emo)
            if label is None:
                unmapped.add(emo)
                label = fallback_label
            labels.append(label)

        text_mat   = np.zeros((T, dim_text),   dtype=np.float32)
        audio_mat  = np.zeros((T, dim_audio),  dtype=np.float32)
        visual_mat = np.zeros((T, dim_visual), dtype=np.float32)

        for i, uid in enumerate(utt_ids):
            text_mat[i] = text_emb[did][uid]
            audio_mat[i] = audio_emb[did][uid]
            v1 = visual_emb1[did][uid]
            visual_mat[i] = v1 if visual_emb2 is None else np.concatenate([v1, visual_emb2[did][uid]])
        if not all(np.isfinite(m).all() for m in (text_mat, audio_mat, visual_mat)):
            raise ValueError(f"Non-finite merged features: {name}")

        videoIDs[name]      = utt_ids
        videoSpeakers[name] = speakers
        videoLabels[name]   = labels
        videoText[name]     = text_mat
        videoAudio[name]    = audio_mat
        videoVisual[name]   = visual_mat
        videoSentence[name] = texts
        testVid.append(name)

        print(f"  [{name}]  visual={visual_mat.shape}")

    if unmapped:
        print(f"  [INFO] Unmapped emotions (→ {label_name} fallback): {sorted(unmapped)}")

    data = [
        videoIDs, videoSpeakers, videoLabels,
        videoText, videoText, videoText, videoText,  # 4 text slots
        videoAudio, videoVisual, videoSentence,
        [],       # trainVid — empty (inference only)
        testVid,
    ]

    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    sample = testVid[0]
    print(f"\nSaved: {output_path}")
    print(f"  {len(testVid)} dialogues  visual_dim={dim_visual}")
    print(f"  [{sample}] audio={data[7][sample].shape}  "
          f"visual={data[8][sample].shape}  labels={data[2][sample]}")


def main():
    import argparse
    from .eval_metrics import phase
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["iemocap-base", "iemocap-full", "meld-base", "meld-full"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with phase("merge"):
        df = pd.read_csv(CSV_PATH, sep=";")
        df.columns = df.columns.str.strip()
        video_files = sorted(df["video_file"].unique())
        video_to_did = {v:i for i,v in enumerate(video_files)}
        df["dialogue_id"] = df["video_file"].map(video_to_did)
        df["utterance_id"] = df.groupby("dialogue_id").cumcount()
        text = load_pkl(TEXT_PKL)
        audio = load_pkl(AUDIO_PKL)
        vit = load_pkl(VISUAL_VIT_PKL)
        secondary = None
        if args.profile == "iemocap-full":
            secondary = load_pkl(VISUAL_LMK_PKL)
        elif args.profile == "meld-full":
            secondary = load_pkl(VISUAL_AU_PKL)
        mapping = _EMOTION_MAP if args.profile.startswith("iemocap") else {
            "neutral":0, "surprise":1, "fear":2, "sadness":3, "sad":3,
            "joy":4, "happy":4, "disgust":5, "anger":6, "angry":6,
        }
        valid = {label for label,value in mapping.items() if value is not None}
        selected = [v for v in video_files if df.loc[df.video_file == v, "emotion"].str.strip().str.lower().isin(valid).all()]
        if not selected:
            raise ValueError("No complete dialogues have supported labels for this checkpoint")
        build_pkl(df, selected, video_to_did, text, audio, vit, secondary,
                  _dim(text), _dim(audio), _dim(vit), _dim(secondary) if secondary else 0,
                  args.output, mapping, 2 if args.profile.startswith("iemocap") else 0, "neutral")


if __name__ == "__main__":
    main()
