"""Save Qwen3-4B-Base vectors at block 18 and the final Topic: token."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
BLOCK = 18  # Count completed transformer blocks, starting at 1.


def make_prompts(stimuli):
    texts, labels = [], []
    for template in stimuli["templates"]:
        for label, name in enumerate(stimuli["disciplines"]):
            texts.append(template.format(discipline=name) + stimuli["suffix"])
            labels.append(label)
    return texts, np.asarray(labels, dtype=np.int32)


def extract(texts, batch_size):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Run extraction on a CUDA GPU; analysis works on a CPU.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    colon = tokenizer(":", add_special_tokens=False)["input_ids"]
    assert len(colon) == 1 and all(ids[-1] == colon[0] for ids in encoded)
    model = AutoModel.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to("cuda").eval()

    # A hook receives the residual-stream output of this block, before final RMSNorm.
    state = {}
    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        state["vectors"] = hidden[batch, state["positions"]].float().cpu().numpy()

    handle = model.layers[BLOCK - 1].register_forward_hook(capture)

    def forward(indices):
        tokens = tokenizer.pad(
            {"input_ids": [encoded[i] for i in indices]},
            padding=True, return_tensors="pt",
        ).to("cuda")
        # Right padding: the last real token is one before the sequence length.
        state["positions"] = tokens["attention_mask"].sum(dim=1) - 1
        with torch.inference_mode():
            model(**tokens, use_cache=False)
        return state["vectors"]

    try:
        short = min(range(len(texts)), key=lambda i: len(encoded[i]))
        long = max(range(len(texts)), key=lambda i: len(encoded[i]))
        alone, padded = forward([short]), forward([short, long])[:1]
        error = float(np.linalg.norm(alone - padded) / np.linalg.norm(alone))
        if error > 1e-4:
            raise RuntimeError(f"Padding changed the readout: relative error {error}")
        print(f"Padding check: {error:.3g}", flush=True)
        started = time.perf_counter()
        chunks = []
        for first in range(0, len(texts), batch_size):
            last = min(first + batch_size, len(texts))
            chunks.append(forward(range(first, last)))
            if first == 0 or last % 160 == 0 or last == len(texts):
                print(f"Extracted {last}/{len(texts)}", flush=True)
        vectors = np.concatenate(chunks)
        if not np.isfinite(vectors).all():
            raise RuntimeError("Nonfinite activations")
        metadata = dict(
            model=MODEL, revision=REVISION, block=BLOCK,
            position="final colon in Topic:", dtype="float32",
            attention="eager", tf32=False, batch_size=batch_size,
            padding_relative_error=error,
            inference_seconds=time.perf_counter() - started,
            gpu=torch.cuda.get_device_name(),
            packages=dict(torch=torch.__version__, transformers=transformers.__version__,
                          numpy=np.__version__),
        )
        return vectors, metadata
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; choose a fresh --out path.")
    if args.out.suffix != ".npz" or args.batch_size < 1:
        parser.error("Use an .npz output and a positive batch size.")
    stimuli = json.loads((ROOT / "stimuli.json").read_text())
    texts, labels = make_prompts(stimuli)
    vectors, metadata = extract(texts, args.batch_size)
    metadata["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ["experiment.py", "stimuli.json", "requirements.txt"]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, activations=vectors, discipline=labels,
                        prompts=np.asarray(texts), names=np.asarray(stimuli["disciplines"]),
                        metadata=np.asarray(json.dumps(metadata)))
    print(f"Saved {vectors.shape} activations to {args.out}")


if __name__ == "__main__":
    main()
