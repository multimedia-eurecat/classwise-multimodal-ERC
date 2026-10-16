"""Run one annotated-input ERC profile and record complete stage times."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
MODULE = "aimara.optimized_evaluation"

PROFILES = {
    "iemocap-base": ("IEMOCAP", "vit", "vit_softmax.pt"),
    "iemocap-full": ("IEMOCAP", "vit_landmarks", "vit_landmarks_class_reliability_dropout01_va_contrastive.pt"),
    "meld-base": ("MELD", "vit", "vit_softmax.pt"),
    "meld-full": ("MELD", "vit_action_units", "vit_action_units_class_reliability_dropout01_va_contrastive.pt"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_annotations(path, video_dir, selected=()):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        if not sample.strip():
            raise ValueError("Empty annotation CSV")
        reader = csv.DictReader(handle, delimiter=";" if ";" in sample.splitlines()[0] else ",")
        reader.fieldnames = [c.strip() for c in reader.fieldnames]
        required = {"video_file", "start", "end", "speaker", "text", "emotion"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing annotation columns: {sorted(required - set(reader.fieldnames))}")
        rows = list(reader)
    if selected:
        unknown = set(selected) - {r["video_file"] for r in rows}
        if unknown:
            raise ValueError(f"Unknown --video values: {sorted(unknown)}")
        rows = [r for r in rows if r["video_file"] in selected]
    if not rows:
        raise ValueError("No annotation rows selected")
    last_start = {}
    for row in rows:
        name = row["video_file"]
        if Path(name).name != name or not (video_dir / name).is_file():
            raise ValueError(f"Missing or invalid video filename: {name}")
        start, end = float(row["start"]), float(row["end"])
        if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
            raise ValueError(f"Invalid interval in {name}: {start}, {end}")
        if start < last_start.get(name, 0):
            raise ValueError(f"Unsorted annotation intervals in {name}")
        last_start[name] = start
        if not row["speaker"].strip() or not row["emotion"].strip():
            raise ValueError(f"Missing speaker or emotion in {name}")
    return rows


def profile_stages(profile, python, au_python, run, checkpoint):
    dataset, visual, _ = PROFILES[profile]
    emb = run / "embeddings"
    result = run / f"results_{profile.replace('-', '_')}.json"
    stages = [
        ("text", [python, "-m", MODULE + ".extract_text"], [emb / "text.pkl"]),
        ("audio", [python, "-m", MODULE + ".extract_audio"], [emb / "audio.pkl", emb / "audio_video_map.pkl"]),
        ("visual_vit", [python, "-m", MODULE + ".extract_visual", "--encoder", "vit"], [emb / "visual_vit.pkl", emb / "visual_preprocessing_cache.pkl"]),
    ]
    if visual == "vit_landmarks":
        stages.append(("visual_landmarks", [python, "-m", MODULE + ".extract_visual", "--encoder", "landmarks"], [emb / "visual_landmarks.pkl"]))
    if visual == "vit_action_units":
        stages.append(("action_units", [au_python, "-m", MODULE + ".extract_action_units"], [emb / "visual_action_units.pkl"]))
    pkl = run / f"evaluation_{profile.replace('-', '_')}.pkl"
    stages.append(("merge", [python, "-m", MODULE + ".merge_pkl", "--profile", profile, "--output", str(pkl)], [pkl]))
    command = [python, "-m", MODULE + ".run_inference", "--checkpoint", str(checkpoint),
               "--dataset", dataset, "--visual-encoder", visual, "--pkl", str(pkl), "--out", str(result)]
    if profile.endswith("full"):
        command += ["--intra-visual-gate", "1"]
    if dataset == "MELD":
        command += ["--temp", "8"]
    stages.append(("erc", command, [result]))
    return stages


def environment_identity(python):
    code = """
import importlib.metadata as m, json, sys, platform
import torch
available = torch.cuda.is_available()
print(json.dumps(dict(python=sys.version, platform=platform.platform(),
    packages=sorted((d.metadata['Name'],d.version) for d in m.distributions()),
    cuda_available=available, cuda_runtime=torch.version.cuda,
    cuda_devices=[dict(name=torch.cuda.get_device_name(i), capability=torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())] if available else [])))
"""
    return json.loads(subprocess.check_output([python, "-B", "-c", code], text=True))


def local_snapshot(model, assets):
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    directory = assets / "huggingface/hub" / ("models--" + model.replace("/", "--"))
    ref = directory / "refs/main"
    if ref.is_file():
        path = directory / "snapshots" / ref.read_text().strip()
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(f"Local model not found: {model}. Supply --text-model/--vit-model directories or prepare the Hugging Face cache under {assets}")


def require_file(path, label):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}; see aimara/optimized_evaluation/README.md")
    return path


def execute_stage(name, command, outputs, env, run, fingerprint, resume, upstream_changed):
    receipt_path = run / "logs" / f"{name}.receipt.json"
    if resume and not upstream_changed and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("fingerprint") == fingerprint and
            all(p.is_file() and p.stat().st_size and receipt["outputs"].get(str(p)) == digest(p) for p in outputs)):
            return dict(name=name, status="reused", wall_seconds=None), False
    metrics = run / "logs" / f"{name}.metrics.json"
    receipt_path.unlink(missing_ok=True)
    metrics.unlink(missing_ok=True)
    stage_env = dict(env, EVAL_STAGE_METRICS=str(metrics))
    started = time.perf_counter()
    with (run / "logs" / f"{name}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=stage_env, cwd=REPO_ROOT, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    row = dict(name=name, status="passed" if code == 0 else "failed",
               wall_seconds=time.perf_counter() - started, command=command, exit_code=code)
    if metrics.exists():
        row.update(json.loads(metrics.read_text()))
    if code or not all(p.is_file() and p.stat().st_size for p in outputs):
        row["status"] = "failed"
        return row, True
    atomic_json(receipt_path, dict(fingerprint=fingerprint, outputs={str(p): digest(p) for p in outputs}))
    return row, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    annotations = parser.add_mutually_exclusive_group(required=True)
    annotations.add_argument("--csv", type=Path)
    annotations.add_argument("--use-predefined-csv", action="store_true")
    parser.add_argument("--input-csv-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/optimized_evaluation")
    parser.add_argument("--run-name")
    parser.add_argument("--video", action="append", default=[], help="Select filename; repeat for several videos")
    parser.add_argument("--assets-root", type=Path, default=REPO_ROOT / ".cache/optimized_evaluation")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--audio-model", type=Path)
    parser.add_argument("--text-model", default="sentence-transformers/all-roberta-large-v1")
    parser.add_argument("--vit-model", default="dima806/facial_emotions_image_detection")
    parser.add_argument("--asd-weights", type=Path)
    parser.add_argument("--landmark-model", type=Path)
    parser.add_argument("--landmark-bfm", type=Path)
    parser.add_argument("--landmark-normalization", type=Path)
    parser.add_argument("--sdt-python", default=sys.executable)
    parser.add_argument("--au-python", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--asd-lookahead", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true", default=os.environ.get("EVAL_RESUME") == "1")
    parser.add_argument("--dry-run", action="store_true", help="Print stage plan without loading models or writing files")
    args = parser.parse_args(argv)
    if not math.isfinite(args.asd_lookahead) or args.asd_lookahead < 0:
        parser.error("--asd-lookahead must be finite and nonnegative")
    args.video_dir = args.video_dir.resolve()
    name = args.run_name or args.video_dir.name
    if Path(name).name != name or name in (".", ".."):
        parser.error("--run-name must be a single directory name")
    run = args.output_root.resolve() / name / args.profile
    dataset, _, checkpoint_name = PROFILES[args.profile]
    checkpoint = (args.checkpoint or args.checkpoint_root / dataset / "final" / checkpoint_name).expanduser().resolve()
    args.au_python = args.au_python or args.sdt_python
    stages = profile_stages(args.profile, args.sdt_python, args.au_python, run, checkpoint)
    if args.dry_run:
        print(json.dumps(dict(profile=args.profile, output=str(run), stages=[dict(name=n, command=c) for n,c,_ in stages]), indent=2))
        return 0
    source = args.csv or (args.input_csv_dir or args.video_dir.parent.parent / "input_csv") / f"{args.video_dir.name}.csv"
    rows = read_annotations(source, args.video_dir, args.video)
    videos = sorted({r["video_file"] for r in rows})
    durations = {}
    for video in videos:
        info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(args.video_dir / video)], text=True))
        duration = float(info["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid media duration: {video}")
        if max(float(r["end"]) for r in rows if r["video_file"] == video) > duration + .05:
            raise ValueError(f"Annotation exceeds video duration: {video}")
        durations[video] = duration
    assets = args.assets_root.expanduser().resolve()
    checkpoint = require_file(checkpoint, "ERC checkpoint")
    audio_model = (args.audio_model or assets / "modelscope/iic/emotion2vec_plus_large").expanduser().resolve()
    dependency_files = [checkpoint] + [require_file(audio_model / name, "emotion2vec asset") for name in ("model.pt", "config.yaml", "configuration.json")]
    asd_weights = require_file(args.asd_weights or assets / "light-asd/pretrain_AVA_CVPR.model", "Light-ASD weights")
    dependency_files.append(asd_weights)
    text_model = local_snapshot(args.text_model, assets)
    vit_model = local_snapshot(args.vit_model, assets)
    landmark_assets = {}
    if args.profile == "iemocap-full":
        for key, value in {
            "EVAL_LANDMARK_MODEL": args.landmark_model or assets / "landmarks/mb1_120x120.onnx",
            "EVAL_LANDMARK_BFM": args.landmark_bfm or assets / "landmarks/bfm_noneck_v3.pkl",
            "EVAL_LANDMARK_NORMALIZATION": args.landmark_normalization or assets / "landmarks/param_mean_std_62d_120x120.pkl",
        }.items():
            path = require_file(value, key)
            landmark_assets[key] = str(path)
            dependency_files.append(path)
    for snapshot in (text_model, vit_model):
        dependency_files.extend(p for p in snapshot.rglob("*") if p.is_file())
    for attribute in ("sdt_python", "au_python"):
        if attribute == "au_python" and args.profile != "meld-full":
            continue
        executable = getattr(args, attribute)
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(f"Runtime interpreter not found: {executable}")
        # Keep the venv executable itself: resolving its symlink would select
        # the base interpreter and lose the environment's installed packages.
        setattr(args, attribute, str(Path(resolved).absolute()))
    stages = profile_stages(args.profile, args.sdt_python, args.au_python, run, checkpoint)
    identities = {args.sdt_python: environment_identity(args.sdt_python)}
    if args.profile == "meld-full":
        identities[args.au_python] = environment_identity(args.au_python)
    requested_device = args.device
    if args.device == "auto":
        args.device = "cuda" if identities[args.sdt_python]["cuda_available"] else "cpu"
    if args.device == "cuda" and any(not value["cuda_available"] for value in identities.values()):
        raise RuntimeError("CUDA requested but unavailable in a selected runtime")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", EVAL_RUN_ROOT=str(run),
               EVAL_VIDEO_DIR=str(args.video_dir), EVAL_CSV_PATH=str(run / "pseudo_gt.csv"),
               EVAL_EMB_DIR=str(run / "embeddings"), EVAL_TEXT_PKL=str(run / "embeddings/text.pkl"),
               EVAL_AUDIO_PKL=str(run / "embeddings/audio.pkl"), EVAL_AU_PKL=str(run / "embeddings/visual_action_units.pkl"),
               EVAL_AUDIO_MODEL=str(audio_model), EVAL_TEXT_MODEL=str(text_model), EVAL_VIT_MODEL=str(vit_model),
               EVAL_ASD_WEIGHTS=str(asd_weights), EVAL_DEVICE=args.device, EVAL_ASD_LOOKAHEAD=str(args.asd_lookahead),
               HF_HOME=str(assets / "huggingface"), HUGGINGFACE_HUB_CACHE=str(assets / "huggingface/hub"),
               HF_HUB_DISABLE_XET="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               MPLCONFIGDIR=str(run / "cache/matplotlib"),
               XDG_CACHE_HOME=str(run / "cache/xdg"), NUMBA_CACHE_DIR=str(run / "cache/numba"),
               PYTHONPATH=str(REPO_ROOT) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
               **landmark_assets)
    source_files = [p for directory in (REPO_ROOT / "aimara", REPO_ROOT / "unimodal", REPO_ROOT / "multimodal")
                    for p in directory.rglob("*.py") if "__pycache__" not in p.parts]
    identity = dict(profile=args.profile, device=args.device, asd_lookahead=args.asd_lookahead,
                    annotations=rows, video_sha256={v: digest(args.video_dir / v) for v in videos},
                    source_sha256={str(p.relative_to(REPO_ROOT)):digest(p) for p in sorted(source_files)},
                    dependencies={str(p):digest(p) for p in dependency_files}, environments=identities,
                    settings={k:v for k,v in env.items() if k in (
                        "EVAL_FACE_BATCH_SIZE", "EVAL_VIT_BATCH_SIZE", "EVAL_AUDIO_MODEL", "EVAL_TEXT_MODEL", "EVAL_VIT_MODEL", "EVAL_ASD_WEIGHTS", "EVAL_DEVICE", "EVAL_ASD_LOOKAHEAD",
                        "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    env["EVAL_FINGERPRINT"] = fingerprint
    previous = run / "configuration.json"
    if previous.exists():
        if not args.resume:
            raise ValueError(f"Output already exists: {run}; choose a fresh --run-name or explicitly --resume")
        if json.loads(previous.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Resume configuration, code, dependencies or inputs changed; use a fresh output directory")
    elif run.exists() and any(run.iterdir()):
        raise ValueError(f"Unrecognized existing output directory: {run}")
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "embeddings").mkdir(exist_ok=True)
    for path in (env["MPLCONFIGDIR"], env["XDG_CACHE_HOME"], env["NUMBA_CACHE_DIR"]):
        Path(path).mkdir(parents=True, exist_ok=True)
    atomic_json(previous, dict(fingerprint=fingerprint, identity=identity))
    with (run / "pseudo_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    report_path = run / "timing_report.json"
    report = dict(schema_version=1, status="running", profile=args.profile,
                  scope="Annotated-input ERC evaluation; excludes VAD, ASR and diarization",
                  requested_device=requested_device, effective_device=args.device,
                  timing_method="Sequential subprocess wall time includes imports, model loading, preprocessing, inference and stage writes; validation, hashing and final report rendering excluded",
                  memory_method="Per-stage parent peak RSS and Torch allocator peaks; excludes FFmpeg child RSS and non-Torch GPU allocations",
                  asd_lookahead_seconds=args.asd_lookahead, media_seconds=sum(durations.values()),
                  videos=durations, fingerprint=fingerprint, stages=[], pipeline_rtf=None)
    atomic_json(report_path, report)
    changed = False
    started = time.perf_counter()
    try:
        for stage_name, command, outputs in stages:
            row, executed = execute_stage(stage_name, command, outputs, env, run, fingerprint, args.resume, changed)
            changed = changed or executed
            report["stages"].append(row)
            atomic_json(report_path, report)
            if row["status"] == "failed":
                raise RuntimeError(f"Stage {stage_name} failed; see {run / 'logs' / (stage_name + '.log')}")
        # A resumed attempt may reuse a preprocessing cache left by a failed
        # stage even when every stage is executed again. It is not a fresh RTF.
        reused = args.resume or any(r["status"] == "reused" for r in report["stages"])
        report.update(status="resumed" if reused else "passed",
                      executed_stage_seconds=sum(r["wall_seconds"] or 0 for r in report["stages"]))
        if not reused:
            report["pipeline_rtf"] = report["executed_stage_seconds"] / report["media_seconds"]
        result = json.loads(stages[-1][2][0].read_text())
        report["accuracy"] = {k:result[k] for k in ("overall_accuracy", "n_correct", "n_total")}
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", pipeline_rtf=None)
        raise
    finally:
        report["evaluation_wall_seconds"] = time.perf_counter() - started
        atomic_json(report_path, report)
    print(f"Report: {report_path}")
    print(f"Profile {args.profile}: status={report['status']}, RTF={report['pipeline_rtf']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
