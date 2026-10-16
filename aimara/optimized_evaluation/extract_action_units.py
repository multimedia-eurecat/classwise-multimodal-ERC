"""Extract AU features using the same active-speaker cache as ViT."""
import os
import pickle
import pandas as pd
import torch
from .au_features import extract_crop_aus, pool_aus
from .eval_metrics import phase, DETAILS, check_embeddings
from .visual_cache import _cache_signature, _read_unique_crops


def main():
    directory = os.environ['EVAL_EMB_DIR']
    df = pd.read_csv(os.environ['EVAL_CSV_PATH'], sep=';')
    df.columns = df.columns.str.strip()
    files = sorted(df.video_file.unique())
    df['dialogue_id'] = df.video_file.map({v:i for i,v in enumerate(files)})
    df['utterance_id'] = df.groupby('dialogue_id').cumcount()
    with open(os.path.join(directory, 'visual_preprocessing_cache.pkl'), 'rb') as handle:
        cache = pickle.load(handle)
    if cache.get('signature') != _cache_signature(df):
        raise ValueError('AU preprocessing cache is missing or stale; run ViT first')
    device = os.environ.get('EVAL_DEVICE', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with phase('model_load'):
        from feat import Detector
        detector = Detector(au_model='svm', emotion_model='svm', device=device)
    embeddings = {}
    successful = 0
    attempted = 0
    for video, rows in df.groupby('video_file', sort=True):
        did = int(rows.iloc[0].dialogue_id)
        values = {int(uid):[] for uid in rows.utterance_id}
        with phase('crop_decode'):
            crops = _read_unique_crops(
                os.path.join(os.environ['EVAL_VIDEO_DIR'], video),
                cache['indices'][video], cache['speaker_boxes'][video], 0.2,
            )
        with phase('encoder_inference'):
            for uids, crop in crops:
                attempted += 1
                vector = extract_crop_aus(crop, detector)
                if vector is not None:
                    successful += 1
                    for uid in uids:
                        values[uid].append(vector)
        embeddings[did] = {uid:pool_aus(frames) for uid,frames in values.items()}
        print(f'[{video}] {len(crops)} crops; {sum(bool(v) for v in values.values())}/{len(values)} utterances with AU detections', flush=True)
    DETAILS.update(attempted_crops=attempted, successful_crops=successful)
    check_embeddings(embeddings, 60)
    with open(os.environ['EVAL_AU_PKL'], 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
