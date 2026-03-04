# translate_npy.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf


def _is_cuda_runtime_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return (
        "cuda" in msg
        or "cudnn" in msg
        or "cublas" in msg
        or "memory allocation failure" in msg
    )


def _cuda_sanity_check() -> Optional[str]:
    """Return None if CUDA seems usable, else an error string."""
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False"
    try:
        # Small allocation catches many driver/runtime issues early.
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def add_repo_to_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root not found: {repo_root}")
    # Ensure we import *this repo's* fairseq first (not pip fairseq)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def ensure_fairseq_dict_for_spm(spm_model: Path) -> Path:
    """
    The SignToTextTask in this repo typically loads a Fairseq dict file from
    the SentencePiece model path by swapping .model -> .txt.
    If it's missing, we auto-generate it from the SPM model.
    """
    spm_model = spm_model.resolve()
    if not spm_model.exists():
        raise FileNotFoundError(f"Missing SPM model: {spm_model}")

    dict_path = Path(str(spm_model).replace(".model", ".txt"))
    if dict_path.exists():
        return dict_path

    # Auto-generate dict
    try:
        import sentencepiece as spm
    except Exception as e:
        raise RuntimeError(
            f"Missing Fairseq dict file {dict_path} and cannot import sentencepiece to create it.\n"
            f"Install sentencepiece, or create the dict manually.\n"
            f"Original error: {e}"
        )

    sp = spm.SentencePieceProcessor(model_file=str(spm_model))

    FAIRSEQ_SPECIALS = {"<pad>", "<s>", "</s>", "<unk>"}
    lines = []
    seen = set()
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece in FAIRSEQ_SPECIALS:
            continue
        if piece in seen:
            continue
        lines.append(f"{piece} 1")
        seen.add(piece)

    dict_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote Fairseq dict for SPM: {dict_path} (size={len(lines)})")
    return dict_path


def _as_omegaconf(cfg_raw: Any) -> Any:
    """
    cfg_raw can be:
      - OmegaConf DictConfig
      - plain dict (common for some fairseq checkpoints)
    Return: DictConfig
    """
    if OmegaConf.is_config(cfg_raw):
        return OmegaConf.create(OmegaConf.to_container(cfg_raw, resolve=True))
    if isinstance(cfg_raw, dict):
        return OmegaConf.create(cfg_raw)
    raise TypeError(f"Unexpected cfg type in checkpoint: {type(cfg_raw)}")


def load_task_and_model(
    ckpt_path: Path,
    repo_root: Path,
    data_dir: Path,
    spm_model: Path,
    device: torch.device,
):
    add_repo_to_path(repo_root)

    # Import after sys.path manipulation
    from fairseq import tasks  # type: ignore

    ckpt_path = ckpt_path.resolve()
    state = torch.load(str(ckpt_path), map_location="cpu")

    cfg_raw = state.get("cfg", None)
    if cfg_raw is None:
        raise RuntimeError("Checkpoint missing 'cfg' (expected Hydra/OmegaConf-style checkpoint).")

    cfg = _as_omegaconf(cfg_raw)

    # Make sure SPM dict exists (repo often expects .txt next to .model)
    _ = ensure_fairseq_dict_for_spm(spm_model)

    # Patch paths to local
    if not hasattr(cfg, "task"):
        raise RuntimeError("Checkpoint cfg missing 'task' section.")
    if not hasattr(cfg, "model"):
        raise RuntimeError("Checkpoint cfg missing 'model' section.")

    cfg.task.data = str(data_dir.resolve())
    # Your repo uses this key (as seen in your training logs)
    cfg.task.bpe_sentencepiece_model = str(spm_model.resolve())

    # If you ever want to disable moses preprocessing:
    # cfg.task.pre_tokenizer = None

    task = tasks.setup_task(cfg.task)

    model = task.build_model(cfg.model)

    # Load weights (try strict first; fallback to non-strict with warnings)
    try:
        model.load_state_dict(state["model"], strict=True)
    except RuntimeError as e:
        print("[WARN] strict=True state_dict load failed; retrying strict=False")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print(f"[WARN] Missing keys: {len(missing)}")
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
        print(f"[WARN] Original error: {e}")

    model.to(device)
    model.eval()

    return cfg, task, model


def make_sample(i3d: np.ndarray, mp: np.ndarray, device: torch.device) -> Dict[str, Any]:
    """
    Build a fairseq-style sample for inference.

    Expected:
      i3d: (T,1024) float32
      mp : (T,33,3) float32/float64
    """
    if i3d.ndim != 2 or i3d.shape[1] != 1024:
        raise ValueError(f"Expected i3d shape (T,1024), got {i3d.shape}")

    if mp.ndim != 3 or mp.shape[1:] != (33, 3):
        raise ValueError(f"Expected mp shape (T,33,3), got {mp.shape}")

    T = int(min(i3d.shape[0], mp.shape[0]))
    if T <= 0:
        raise ValueError("Empty sequence after aligning i3d and mp lengths.")

    i3d = i3d[:T].astype(np.float32, copy=False)
    mp = mp[:T].astype(np.float32, copy=False)

    # (B,T,C) and (B,T,33,3)
    src_tokens = torch.from_numpy(i3d).unsqueeze(0).to(device)
    src_mp = torch.from_numpy(mp).unsqueeze(0).to(device)

    encoder_padding_mask = torch.zeros((1, T), dtype=torch.bool, device=device)  # False = not padded
    mediapipe_padding_mask = torch.zeros((1, T), dtype=torch.bool, device=device)

    sample = {
        "id": torch.LongTensor([0]).to(device),
        "net_input": {
            "src_tokens": src_tokens,
            "src_mediapipe_tokens": src_mp,
            "encoder_padding_mask": encoder_padding_mask,
            "mediapipe_padding_mask": mediapipe_padding_mask,
        },
    }
    return sample


def decode_hypo(task, tokens: torch.Tensor) -> str:
    """
    Convert token IDs -> text using task dictionary and (if available) task.bpe (SentencePiece).
    """
    # Fairseq dict -> "pieces"
    s = task.target_dictionary.string(tokens)

    # If fairseq built a BPE decoder (SentencePiece), use it
    bpe = getattr(task, "bpe", None)
    if bpe is not None:
        try:
            s = bpe.decode(s)
            return " ".join(str(s).split())
        except Exception:
            pass

    # Fallback: basic sentencepiece "▁" handling
    s = str(s).replace("▁", " ").strip()
    s = " ".join(s.split())
    return s


@torch.no_grad()
def translate_one(task, model, i3d_path: Path, mp_path: Path, beam: int, max_len_b: int) -> str:
    i3d = np.load(str(i3d_path))
    mp = np.load(str(mp_path))

    device = next(model.parameters()).device

    def _run_one(curr_device: torch.device) -> str:
        sample = make_sample(i3d, mp, curr_device)

        # Build generation cfg (don’t rely on checkpoint yaml being perfect)
        gen_cfg = OmegaConf.create(
            {
                "beam": int(beam),
                "max_len_a": 0.0,
                "max_len_b": int(max_len_b),
                "lenpen": 1.0,
                "no_repeat_ngram_size": 2,
            }
        )

        generator = task.build_generator([model], gen_cfg)
        hypos = task.inference_step(generator, [model], sample)
        best = hypos[0][0]
        hypo_tokens = best["tokens"].detach().cpu()
        return decode_hypo(task, hypo_tokens)

    try:
        return _run_one(device)
    except RuntimeError as e:
        if device.type == "cuda" and _is_cuda_runtime_error(e):
            print(f"[WARN] CUDA inference failed ({e}); retrying on CPU.")
            model.to("cpu")
            return _run_one(torch.device("cpu"))
        raise


def _resolve_path(base: Path, p: Any) -> Path:
    """
    Resolve paths from TSV that might be absolute or relative.
    """
    pp = Path(str(p))
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--spm_model", required=True, type=Path)
    ap.add_argument("--data_dir", required=True, type=Path, help="Any existing directory (used to patch cfg.task.data)")
    ap.add_argument("--pairs_tsv", type=Path, default=None, help="Optional TSV with columns: id, i3d, mp, translation(optional)")
    ap.add_argument("--i3d", type=Path, default=None, help="Single i3d npy to translate")
    ap.add_argument("--mp", type=Path, default=None, help="Single mp npy to translate")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_len_b", type=int, default=40)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        cuda_err = _cuda_sanity_check()
        if cuda_err is not None:
            print(f"[WARN] CUDA sanity check failed ({cuda_err}); falling back to CPU.")
            device = torch.device("cpu")

    # sanity checks
    for p in [args.repo_root, args.ckpt, args.spm_model, args.data_dir]:
        if not p.exists():
            raise FileNotFoundError(p)

    # Ensure SPM dict exists next to model (auto-create if missing)
    ensure_fairseq_dict_for_spm(args.spm_model)

    try:
        _, task, model = load_task_and_model(
            args.ckpt, args.repo_root, args.data_dir, args.spm_model, device
        )
    except RuntimeError as e:
        if device.type == "cuda" and _is_cuda_runtime_error(e):
            print(f"[WARN] CUDA model setup failed ({e}); retrying on CPU.")
            device = torch.device("cpu")
            _, task, model = load_task_and_model(
                args.ckpt, args.repo_root, args.data_dir, args.spm_model, device
            )
        else:
            raise

    if args.pairs_tsv is not None:
        import pandas as pd

        pairs_tsv = args.pairs_tsv.resolve()
        if not pairs_tsv.exists():
            raise FileNotFoundError(pairs_tsv)

        df = pd.read_csv(pairs_tsv, sep="\t")
        need_cols = {"id", "i3d", "mp"}
        if not need_cols.issubset(set(df.columns)):
            raise ValueError(f"pairs_tsv must contain columns {need_cols}, got {list(df.columns)}")

        base = pairs_tsv.parent
        for _, row in df.iterrows():
            _id = str(row["id"])
            i3d_path = _resolve_path(base, row["i3d"])
            mp_path = _resolve_path(base, row["mp"])

            if not i3d_path.exists() or not mp_path.exists():
                print(f"[SKIP] {_id}: missing file(s)\n  i3d={i3d_path}\n  mp={mp_path}")
                continue

            pred = translate_one(task, model, i3d_path, mp_path, args.beam, args.max_len_b)
            ref = str(row["translation"]) if "translation" in df.columns else ""
            print(f"\n=== {_id} ===")
            if ref and ref != "nan":
                print("REF:", ref)
            print("HYP:", pred)
        return

    # single example mode
    if args.i3d is None or args.mp is None:
        raise ValueError("Provide either --pairs_tsv OR both --i3d and --mp")

    pred = translate_one(task, model, args.i3d, args.mp, args.beam, args.max_len_b)
    print(pred)


if __name__ == "__main__":
    main()