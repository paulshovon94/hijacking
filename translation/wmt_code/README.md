# Translation-task extension (De→En)

Shows the hyperparameter-stealing pipeline works on **WMT16 De→En translation**, not just
summarization.

The claim is about the *pipeline*, so the shadow models only need to be capable
translators — they deliberately do **not** mirror the summarization model zoo. The covert
sentiment task, the word-substitution attack, the seven feature modalities, the
classifier, and the LoRA grid are all unchanged; only the task and the models differ.

## Read this first: the column names lie, deliberately

The training JSON keeps the summarization pipeline's format verbatim:

```json
{"summarization": [{"real": "<German source>", "summarize": "<English target>"}]}
```

`real` holds **German**, `summarize` holds **English**. The poison CSV likewise keeps the
column name `real_dataset` while holding German text.

This is intentional. Keeping the names means every dataset loader, every feature
extractor's dataframe access, and the entire dataloader/classifier stage work with **no
edits at all**. The only training-side difference from the summarization pipeline is the
prompt string. Renaming the fields would have meant touching a dozen files to achieve
nothing.

## What differs from `summarization/imdb_code/`

| Piece | Change |
|---|---|
| Encoder-decoder prompt | `"summarize: "` → `"translate German to English: "` |
| Decoder-only prompt | `f"Text: {text}\nSummary:"` → `f"German: {text}\nEnglish:"` |
| `max_source_length` | 512/1024 → **256** (see below) |
| Cover task | CNN/DailyMail → WMT16 De→En news |
| Poison source | New: German side built by MT (see below) |
| Everything else | Unchanged — features x1–x7, classifier, LoRA grid |

**Why 256.** The trainers pad to `max_length`. WMT16 pairs are ~30–60 tokens, so the
inherited 512/1024 would waste most of every batch across 540 runs. This is a data-shape
setting, not a hyperparameter under test, and it applies uniformly to every config.

**Features stay identical, including ROUGE.** BLEU/chrF are reported by
`evaluate_translation.py` as *cover-task quality metrics only*. Feeding them to the
classifier would change the feature space and confound the comparison. If you later want
a BLEU feature ablation, extraction saves `texts_batch_*.json` with both the generated
output and the reference, so it can be computed without re-running any shadow model.

## The poison

Reuses `summarization/transformed_data/imdb/hijacking_imdb.csv` **unchanged** — the covert
task artifacts are byte-identical to the summarization experiment. Stages 1–3 of the
original pipeline are not re-run.

The one new piece is the German source. The summarization poison pairs
(IMDB review → hijacked summary) look like legitimate *summarization* pairs, which is what
lets them blend into CNN/DailyMail. The translation analog needs pairs that look like
legitimate *translation* pairs, and there is no German IMDB. So
`translate_poison_source.py` machine-translates each PEGASUS pseudo-summary En→De with
`Helsinki-NLP/opus-mt-en-de` and pairs it with the already-hijacked English text. Source
and target are then the same content at the same length; the only anomaly is the
substituted stop words.

**Poison rate.** Full poisoning: `POISON_TRAIN_RATIO = 1.0`, so all 9,645 hijacked pairs
go into training against 287,113 WMT pairs — a **~3.25%** training poison rate.
`test.json` therefore holds only the clean WMT validation set, which is what the
cover-task quality check wants; feature extraction probes the models with
`hijacking_wmt.csv` directly, not with `test.json`, so nothing downstream needs poison in
the validation split.

Note this differs from the summarization run, whose `prepare_json_data.py` carries
`split_ratio=.3` (~1.0% effective). The translation experiment uses the full poison pool
deliberately.

**Limitation to state in the paper:** the German source is machine-generated, so it
carries MT artifacts the clean WMT data does not. A defender profiling source-side
fluency could potentially separate poison from clean. This does not affect the stealing
result.

## Pipeline

Run from this directory, with the conda env active and caches pointed at `/work`.

```bash
# 1. Cover task: WMT16 De-En, subsampled to CNN/DailyMail's size
python prepare_wmt16_deen.py

# 2. Poison: build the German source side (reads the summarization poison, never writes it)
python translate_poison_source.py

# 3. Combine into training JSON
python prepare_json_data.py

# 4. Configs: 540 = 10 checkpoints x 54 LoRA combinations
python generate_configs_lora.py

# 5. VIABILITY GATE -- do this before the full sweep (see below)
python prepare_json_data.py --limit 20000 --suffix _smoke
python generate_configs_lora.py --smoke
sbatch --export=ALL,CODE_DIR=$PWD slurm/smoke_test.sbatch

# 6. Full sweep, inside tmux on the login node
tmux new -s sweep
./slurm/run_sweep.sh

# 7. Behavioral features x1-x7, per family
python create_model_features_lora.py            --model_indices 0-107     # BART
python create_model_features_pegasus_lora.py    --model_indices 108-215   # Pegasus
python create_model_features_gpt2_lora.py       --model_indices 216-377   # GPT-2
python simple_create_model_features_phi_lora.py --model_indices 378-431   # Phi
python create_model_features_llama3-1_lora.py   --model_indices 432-485   # LLaMA
python create_model_features_qwen2-5_lora.py    --model_indices 486-539   # Qwen

# 8. Aggregate and train the attack classifier
python create_dataloader_lora.py
torchrun --nproc-per-node=1 experiment_lora.py --seed 42
```

`--model_indices` ranges map to families; `python sweep.py index <family> 0` prints the
first index for any family rather than relying on the table above staying current.

## The viability gate, and what it ruled out

The gate trains one short config per checkpoint (1 epoch on 20k pairs, lr 1e-4, r=8,
alpha=16 — uniform across families) and scores BLEU/chrF on clean WMT16 test. It is cheap
insurance before committing 54 runs per checkpoint.

The first round tested the summarization zoo and ruled most of it out:

| Model | BLEU | chrF | Verdict |
|---|---|---|---|
| BART-large | 23.97 | 47.16 | kept |
| Qwen2.5-7B | 23.47 | 59.15 | kept |
| LLaMA-3.1-8B | 11.27 | 46.14 | kept |
| BART-base | 5.22 | 22.44 | dropped |
| Pegasus-xsum | 3.96 | 20.52 | dropped |
| Phi-1.5 | 1.74 | 21.51 | dropped |
| GPT-2 large / medium / small | 1.24 / 1.02 / 0.64 | ~15–19 | dropped |
| Pegasus-large | 1.14 | 18.54 | dropped |

GPT-2, Pegasus and Phi are English-only pretrained; byte-level BPE lets them *encode*
German but they have no German competence to build on. Stealing hyperparameters from
models that cannot perform the task would undercut the result, so they were replaced with
models built for translation rather than rescued.

The trainers and extractors for the dropped families are still in this directory
(`*_gpt2_*`, `*_pegasus_*`, `*_phi_*`) so the gate result stays reproducible, but nothing
in `sweep.py` references them.

## Labels

Six heads, matching `experiment_lora.py`: `model_family`, `model_size`, `learning_rate`,
`lora_r`, `lora_alpha`, `lora_dropout`. Optimizer and batch size are fixed in the LoRA
grid (`adamw`, 4), so they are not predicted.

One checkpoint per family — **216 runs** (4 × 54):

| Family | Checkpoint | Gate BLEU | ~Time/model |
|---|---|---|---|
| Marian | `Helsinki-NLP/opus-mt-de-en` | 42.50 | ~2 h |
| BART | `facebook/bart-large` | 24.11 | ~4 h |
| Qwen2.5 | `Qwen/Qwen2.5-1.5B` | 40.30 | ~9 h |
| LLaMA | `meta-llama/Llama-3.2-1B` | 15.29 | ~5 h |

Sizes were chosen by gate BLEU, not parameter count. Bigger was not better at the gate
budget — Qwen-1.5B beat both 0.5B (30.09) and 7B (23.47), and Llama-3.2-1B beat 3B (10.62)
and 8B (11.15) — so the best performers are also the cheaper ones. Total ≈ 1,100 GPU-hours,
about 6 days at 8 concurrent tasks.

Marian is purpose-built for De→En and needs no new trainer: it is an encoder-decoder whose
attention projections are named `q_proj`/`v_proj`, which is exactly what
`train_shadow_models_lora.py` already targets. It is also the strongest translator here
*and* the cheapest to train.

**`model_size` is not an independent result in this setup.** With one checkpoint per
family, size is fully determined by family, so those two heads carry identical
information. Report `model_family`, `learning_rate`, `lora_r`, `lora_alpha` and
`lora_dropout` as the real heads; a high `model_size` accuracy here says nothing beyond
what `model_family` already says.

## Comparison baseline

The comparison must be against **summarization LoRA**, not summarization full
fine-tuning — LoRA vs. full FT would be a second variable.

The existing summarization LoRA run covers BART only
(`create_dataloader_only_bart_lora.py` filters to it and sets a single-class family head).
To make the comparison like-for-like, the same 54-config grid needs running on the
summarization data for GPT-2, Phi, LLaMA, and Qwen as well — 324 additional runs. BART and
Pegasus are already done.

## Bugs fixed here, still present in `summarization/imdb_code/`

Both were found while porting and are worth fixing on the summarization side too:

1. **LoRA hyperparameters were ignored by the decoder-only trainers.**
   `train_shadow_model_llama3-1_lora.py`, `train_shadow_model_qwen2-5_lora.py`, and
   `simple_shadow_models_phi_lora.py` hardcoded `r=8, lora_alpha=16, lora_dropout=0.05`
   instead of reading them from the config. All 54 configs for those families would have
   trained identically apart from learning rate, making three of the six heads
   meaningless. The BART trainer always read them correctly.

2. **Train/inference prompt mismatch.** The decoder-only trainers used
   `f"Text: {text}\nSummary:"` (newline) while the matching feature extractors used
   `f"Text: {text} Summary:"` (space). Here both sides use the same prompt.

## Cluster notes

QB4 login nodes have no GPUs, so training goes through SLURM. `slurm/run_sweep.sh` runs on
the login node inside tmux, submits one job array per family, polls `squeue`, and reports
failed tasks at the end. Running the sweep inside `salloc` would die at the 72h walltime
cap, which 540 runs will exceed.

- Account `loni_llmsecull`, partition `gpu2`, one A100 per array task (`-n 32`).
- The 4-node cap allows 8 concurrent single-GPU tasks, hence the `%8` array throttle.
- All caches under `/work/shovon/LLM/`; `/home` is 10 GB and nearly full.
- Conda env expected at `/work/shovon/.conda/envs/hijacking` (override with `CONDA_ENV`).
- `sacrebleu` is an additional dependency, used only by `evaluate_translation.py`.

Dry-run the sweep before committing GPU hours:

```bash
DRY_RUN=1 ./slurm/run_sweep.sh
```
