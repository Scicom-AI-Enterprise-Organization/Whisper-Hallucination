#!/usr/bin/env python3
"""Fine-tune whisper-large-v3 to hallucinate less WITHOUT losing accuracy.

Both halves matter and they pull against each other: a model that emits nothing scores
perfectly on the non-speech arms and is useless. So the target for a blank clip is the empty
transcript, the target for a positive clip is its text, and the mix ratio between them is the
sweep's main factor (`train/build_mix.py`).

  --method lora   PEFT adapters on the attention projections AND the MLP (fc1/fc2, where
                  most of a Whisper block's parameters live). Cheap enough to sweep.
  --method full   every weight. One or two runs, for the mix that wins the sweep.

Target format follows the same contract as the rest of the Whisper family:

    <|startoftranscript|><|LANG|><|transcribe|><|notimestamps|> TEXT<|endoftext|>

A blank clip gets that prefix with no TEXT, which is what teaches "this audio has no words"
rather than "emit something short". Prompt tokens are masked to -100 so loss falls only on the
text and the final EOT.

    .venv_train/bin/python train/finetune_whisper.py --mix train/mixes/plus_synth.jsonl \
        --method lora --out runs/lora_plus_synth
"""
import argparse, io, json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SR = 16000


class ClipDataset(torch.utils.data.Dataset):
    """Reads a mix jsonl. Audio is loaded lazily -- the corpus is 111 h and does not fit in RAM."""

    def __init__(self, rows, processor, max_label_len=200, default_lang="en"):
        self.rows = rows
        self.proc = processor
        self.tok = processor.tokenizer
        self.feat = processor.feature_extractor
        self.max_label_len = max_label_len
        self.default_lang = default_lang

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import soundfile as sf
        r = self.rows[i]
        try:
            x, sr = sf.read(r["audio_filepath"], dtype="float32")
        except Exception:
            return None
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        feats = self.feat(x, sampling_rate=SR, return_tensors="np").input_features[0]

        lang = (r.get("lang") or "").strip() or self.default_lang
        try:
            self.tok.set_prefix_tokens(language=lang, task="transcribe", predict_timestamps=False)
        except Exception:                      # a language Whisper does not know
            self.tok.set_prefix_tokens(language=self.default_lang, task="transcribe",
                                       predict_timestamps=False)
        ids = self.tok(r.get("text") or "", add_special_tokens=True).input_ids
        if len(ids) > self.max_label_len:
            ids = ids[: self.max_label_len - 1] + [self.tok.eos_token_id]
        # Loss on the transcript only: the four prefix tokens are given, not predicted.
        n_prefix = len(self.tok.prefix_tokens) if hasattr(self.tok, "prefix_tokens") else 4
        labels = [-100] * min(n_prefix, len(ids)) + ids[min(n_prefix, len(ids)):]
        return {"input_features": feats, "labels": labels}


@dataclass
class Collator:
    processor: Any

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        feats = torch.tensor(np.stack([b["input_features"] for b in batch]))
        maxlen = max(len(b["labels"]) for b in batch)
        pad = self.processor.tokenizer.pad_token_id
        labels, attn = [], []
        for b in batch:
            n = maxlen - len(b["labels"])
            labels.append(b["labels"] + [-100] * n)
            attn.append([1] * len(b["labels"]) + [0] * n)
        return {"input_features": feats,
                "labels": torch.tensor(labels, dtype=torch.long),
                "decoder_attention_mask": torch.tensor(attn, dtype=torch.long)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=Path("corpus/val.jsonl"))
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", choices=["lora", "full"], default="lora")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                    help="attention projections AND the MLP. fc1/fc2 hold most of the "
                         "parameters in a Whisper block, so attention-only adapters leave the "
                         "bulk of each layer untouched.")
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 2e-4 lora, 1e-5 full. 1e-3 diverges on this data -- the "
                         "corpus mix hit loss 9.76 there and its checkpoint hallucinates on "
                         "100%% of silence.")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=0, help="0 = only at the end")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT",
                                                              "whisper-hallucination"))
    ap.add_argument("--wandb-group", default=None,
                    help="default: the mix name. Grouping by mix is what makes the "
                         "with-synth / without-synth pair readable in one view.")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # W&B only if there is a key; a sweep must not die because a token is missing.
    use_wandb = not args.no_wandb and bool(os.environ.get("WANDB_API_KEY"))
    wb = None

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    rows = [json.loads(l) for l in args.mix.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    random.Random(args.seed).shuffle(rows)
    blanks = sum(1 for r in rows if not (r.get("text") or "").strip())
    print(f"[train] {args.mix.name}: {len(rows)} clips, {blanks/len(rows):.0%} blank, "
          f"method={args.method}", flush=True)

    if use_wandb:
        import wandb
        wb = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or args.mix.stem,
            job_type=args.method,
            name=args.out.name,
            config={"mix": args.mix.stem, "method": args.method,
                    "lora_r": args.lora_r if args.method == "lora" else None,
                    "lora_alpha": args.lora_alpha if args.method == "lora" else None,
                    "lora_target_modules": (list(args.lora_target_modules)
                                            if args.method == "lora" else None),
                    "steps": args.steps, "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "effective_batch": args.batch_size * args.grad_accum,
                    "base_model": args.model, "n_clips": len(rows),
                    "blank_share": round(blanks / len(rows), 4),
                    "has_lexicon_synth": args.mix.stem in ("all", "plus_synth",
                                                           "balanced", "synth_heavy")},
            tags=[args.mix.stem, args.method], reinit=True)

    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                              lora_dropout=args.lora_dropout, bias="none",
                              target_modules=list(args.lora_target_modules))
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
    else:
        model.gradient_checkpointing_enable()

    lr = args.lr if args.lr is not None else (2e-4 if args.method == "lora" else 1e-5)
    if wb is not None:
        wb.config.update({"lr": lr}, allow_val_change=True)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_steps=args.warmup,
        max_steps=args.steps,
        bf16=True,
        logging_steps=25,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or args.steps,
        report_to=["wandb"] if use_wandb else [],
        run_name=args.out.name,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=ClipDataset(rows, proc),
        data_collator=Collator(proc),
    )
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.method == "lora":
        model.save_pretrained(str(args.out / "adapter"))
        # Merge so the benchmark harness can load it like any other checkpoint.
        merged = model.merge_and_unload()
        merged.save_pretrained(str(args.out / "merged"), safe_serialization=True)
        proc.save_pretrained(str(args.out / "merged"))
    else:
        model.save_pretrained(str(args.out / "merged"), safe_serialization=True)
        proc.save_pretrained(str(args.out / "merged"))
    (args.out / "run.json").write_text(json.dumps({
        "mix": str(args.mix), "method": args.method, "lr": lr, "steps": args.steps,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "lora_r": args.lora_r if args.method == "lora" else None,
        "lora_target_modules": (list(args.lora_target_modules)
                                if args.method == "lora" else None),
        "lora_target_modules": (list(args.lora_target_modules)
                                if args.method == "lora" else None),
        "n_clips": len(rows), "blank_share": round(blanks / len(rows), 4),
        "base_model": args.model,
        # The scorer reopens this run to attach the benchmark numbers, so training curve and
        # final metrics live on one W&B run instead of two unlinked halves.
        "wandb_run_id": wb.id if wb is not None else None,
        "wandb_project": args.wandb_project if wb is not None else None,
    }, indent=2))
    if wb is not None:
        wb.finish()
    print(f"[train] saved -> {args.out}/merged", flush=True)


if __name__ == "__main__":
    main()
