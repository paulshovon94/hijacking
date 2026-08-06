# Translation-task replication (De→En)

Replicates the hyperparameter-stealing attack on **WMT16 De→En translation** to show the
result is not specific to summarization.

The governing principle is **change the task, hold everything else fixed**: same covert
sentiment task, same word-substitution attack, same seven feature modalities, same
classifier, same LoRA grid. Any difference in per-head accuracy is then attributable to
the task and nothing else.

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

**Poison rate.** 9,645 poison rows against 287,113 WMT pairs, of which
`POISON_TRAIN_RATIO = 0.3` goes into training — an effective training poison rate of
~1.0%. This mirrors the summarization run, whose `prepare_json_data.py` carries
`split_ratio=.3`.

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

## The viability gate

Three of the six families were pretrained on English only, and De→En is a real ask for
them. **Pegasus is the likeliest failure** — its SentencePiece vocab is English-trained
and `pegasus-xsum` is already specialized to summarization.

The gate trains one short config per checkpoint on a truncated dataset and scores BLEU on
clean WMT16 test. Six cheap runs decide which families are worth 54 runs each.

Any family that cannot clear the quality threshold is **dropped and reported**, with the
random baseline for the `model_family` head adjusted to the surviving class count.
"The attack applies to models capable of the task" is a defensible scope statement; the
alternative — stealing hyperparameters from models that cannot translate — is not.

## Labels

Six heads, matching `experiment_lora.py`: `model_family`, `model_size`, `learning_rate`,
`lora_r`, `lora_alpha`, `lora_dropout`. Optimizer and batch size are fixed in the LoRA
grid (`adamw`, 4), so they are not predicted.

The ten checkpoints match `configs/config_summary.csv` exactly: BART base/large, Pegasus
xsum/large, GPT-2 small/medium/large, Phi-1.5, LLaMA-3.1-8B, Qwen2.5-7B.

**Known confound, inherited deliberately:** Phi, LLaMA, and Qwen have one size each, so
`model_size` is partly determined by `model_family`. The summarization experiment has the
same property; "fixing" it here would break the comparison.

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
