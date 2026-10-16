# fugojev

**Fu**dan University **G**ynecologic **O**ncology **Jev** — a [MedJev](https://github.com/JunMa11/MedJev)-style decision model built specifically for gynecologic oncology.

The goal is fast FIGO staging of gynecologic cancers from clinical notes on a CPU or a small GPU: a sub-1B frozen backbone, a 42 MB adapter, and closed-set answers with probabilities instead of generated text. **Version 1.2.0** (tag `v1.2.0`) solves endometrial cancer, FIGO 2009; cervical and ovarian cancer are planned for later versions. 1.1 added the reference systems (rule baseline, zero-shot probe), calibration, a runtime benchmark and a JSON request API; 1.2 adds five-fold cross-validation and input ablations, which revise the headline: on this centre's templated pathology the rule baseline stages as well as or better than the model, and the model's advantage appears only when the template is removed. The shipped weights are unchanged since 1.0.

Given the pathology, imaging, outpatient, admission and operative notes of one patient, the model answers five multiple-choice questions: FIGO stage, depth of myometrial invasion, lymphovascular space invasion, lymph-node extent, and histologic grade.

## How it works

A frozen `Qwen/Qwen3.5-0.8B-Base` backbone with a LoRA adapter (r=8, 5.9M trainable parameters) encodes the notes once. Each question is appended as a separate branch that can see the notes but not the other questions, and a pointer head scores the option tokens; softmax over the allowed options gives the answer distribution. If the myometrial-invasion head says ≥1/2, a FIGO prediction of IA is lifted to IB. Training: 4 epochs, AdamW 1e-4, bf16 autocast, about 50 minutes on one RTX 4090.

## Results

321 patients with notes, split by hashed admission number into train 257 / dev 36 / test 28. The test set has not been opened.

### Five-fold cross-validation (293 patients)

The 293 non-test patients are hashed into five folds (57 / 60 / 69 / 57 / 50). One model per fold, trained on the other four with the recipe above but 3 epochs fixed in advance (no epoch selection on the held-out fold); the rule baseline's defaults are the training-fold majority. Rows from the five held-out folds are pooled; 95% CIs are patient-level bootstrap (2000 resamples), the p-value is an exact two-sided McNemar test on discordant patients. `reports/cv/cv-summary.md` has every ablation; `python -m fugojev.cv report` regenerates it.

| Question | fugojev 0.8B | rule baseline | Δ [95% CI] | McNemar p | majority |
|---|---|---|---|---|---|
| FIGO stage (n=292) | 0.839 [0.798, 0.880] | **0.884** [0.846, 0.918] | −0.045 [−0.086, −0.003] | 0.053 | 0.630 |
| Myometrial invasion (n=288) | 0.990 | 0.993 | −0.003 | 1.00 | 0.580 |
| Lymph-node extent (n=283) | 0.965 | 0.979 | −0.014 | 0.29 | 0.890 |
| LVSI (n=271) | 0.996 | 0.993 | +0.004 | 1.00 | 0.742 |
| Grade (n=267) | 0.978 | 0.963 | +0.015 | 0.39 | 0.423 |
| micro (1401 questions) | 0.952 | 0.961 | | | |

Per-fold FIGO accuracy is 0.84 / 0.72 / 0.94 / 0.83 / 0.86 for the model and 0.95 / 0.80 / 0.88 / 0.90 / 0.90 for the rules, so the dev figure below (0.972 on 36 patients) is at the lucky end of the spread. The four component questions are at ceiling for both systems. The model loses FIGO on the rare stages that are defined by a composition rule with only a handful of training examples: stage II (cervical stroma) 6/17 vs 12/17, IIIA 0/5 vs 3/5, IIIC2 1/8 vs 5/8; IA, IB and IVB are equal. Composing the stage from the model's own component heads instead of its FIGO head recovers only 0.3 points, because the missed stages (II, IIIA, IIIB) have no component question.

### Input ablations (same folds, same models)

The held-out notes are rewritten before scoring; training is untouched. `paraphrase` turns every templated pathology cell (`肌层浸润: 浸润深度<1/2肌层` …) into one of several prose sentences chosen deterministically per patient, keeping the evidence but changing its surface form. `strip-template` deletes the cells, which removes the evidence (the word 肌层 then survives in 3% of records, 脉管 and grade in none). `pathology-only` keeps only the pathology section.

| Ablation | Question | fugojev 0.8B | rule baseline | Δ [95% CI] | McNemar p |
|---|---|---|---|---|---|
| paraphrase | Myometrial invasion | **0.965** | 0.747 | +0.219 [0.168, 0.270] | <0.001 |
| | LVSI | **0.993** | 0.867 | +0.125 [0.085, 0.168] | <0.001 |
| | Lymph-node extent | **0.954** | 0.890 | +0.064 [0.031, 0.099] | 0.001 |
| | Grade | 0.805 | **0.921** | −0.116 [−0.169, −0.060] | <0.001 |
| | FIGO stage | 0.743 | 0.764 | −0.021 [−0.085, 0.041] | 0.59 |
| strip-template | FIGO stage | 0.360 | 0.644 | −0.284 | <0.001 |
| pathology-only | FIGO stage | 0.849 | 0.887 | −0.038 [−0.075, 0.000] | 0.09 |

Read together: the rules match the template, the model reads the text. When the same facts are stated in prose the rules drop 10–25 points on the three pathology components and the model stays within 1–3 points of its template score; grade is the exception, because the prose variants of 中分化/低分化 were not in training and the rules still catch the `G` letter. With the evidence removed both systems fall to or below the majority floor, so neither is guessing from the rest of the record. Imaging and operative notes add under one point over pathology alone.

### Development set (36 patients)

Used for epoch selection, so not a final score. Accuracy (macro-F1); `python -m fugojev.compare` produces this table from the reports in `reports/dev/`:

| Question | fugojev 0.8B | rule baseline | zero-shot 0.8B | fugojev 0.5B | majority |
|---|---|---|---|---|---|
| FIGO stage (9, n=36) | **0.972** (0.982) | 0.917 (0.905) | 0.083 (0.062) | 0.750 (0.318) | 0.611 |
| Myometrial invasion (3, n=34) | **1.000** (1.000) | 1.000 (1.000) | 0.382 (0.243) | 0.912 (0.837) | 0.618 |
| Lymph-node extent (4, n=36) | 0.972 (0.929) | **1.000** (1.000) | 0.111 (0.129) | 0.889 (0.471) | 0.889 |
| LVSI (2, n=31) | **1.000** (1.000) | 1.000 (1.000) | 0.806 (0.446) | 0.871 (0.713) | 0.839 |
| Grade (3, n=32) | **1.000** (1.000) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 0.500 |
| micro (169 questions) | **0.988** | 0.982 | 0.456 | 0.882 | |
| latency per patient (4090) | 233 ms | — | 416 ms | 89 ms | |

- *rule baseline* (`fugojev.baseline`): regular expressions over the templated pathology fields (`肌层浸润:`, `脉管侵犯:（-）`, `淋巴结转移情况:总数（0/24）` …), composed into a stage; patterns tuned on train only. On this centre's templated reports it is the system to beat, and under cross-validation the model does not beat it.
- *zero-shot* (`fugojev.base_probe`): the same base model, no adapter, answer read from next-token letter logits. Below the majority floor on every question except grade, so the score comes from the training, not the backbone alone.
- *fugojev 0.5B*: same recipe on Qwen2.5-0.5B.

Calibration (`fugojev.calibrate`): the raw probabilities are already calibrated on dev (FIGO ECE 0.078, NLL 0.183; every other question ECE < 0.02). Leave-one-out temperature + prior fitting does not improve FIGO NLL on 36 patients, so no calibration file is shipped; the tool is there for a larger set.

Runtime (`fugojev.bench_runtime`): on an RTX 4090 in bf16, reading the notes costs 228 ms and each additional question about 1.5 ms (one question 229 ms, all five 234 ms, p95 243 ms); latency is flat from 1k to 3k state tokens; peak memory 1.9 GB. On CPU (fp32, 80 cores) about 6 s per patient with the same answers.

## Install

```bash
conda create -n fugojev python=3.12 -y && conda activate fugojev
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[qwen35]" --upgrade-strategy only-if-needed
```

## Download the base model

The repository ships only the LoRA adapter and pointer head (`weights/ec-v1`, 42 MB). Download the backbone (about 1.7 GB) into `models/`:

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3.5-0.8B-Base --local-dir models/Qwen3.5-0.8B-Base
```

Older `huggingface_hub` versions use `huggingface-cli download` with the same arguments. Behind a restricted network, set `HF_ENDPOINT=https://hf-mirror.com` first, or use ModelScope: `modelscope download --model Qwen/Qwen3.5-0.8B-Base --local_dir models/Qwen3.5-0.8B-Base`.

## Predict

```bash
python -m fugojev.predict --run weights/ec-v1 --base models/Qwen3.5-0.8B-Base --state note.txt
```

`note.txt` holds the five sections in order, `[门诊] [入院] [影像] [病理] [手术]`, each with its conclusion at the end. Or pass the sections as JSON and let the package render them exactly as the training data was rendered:

```bash
python -m fugojev.predict --run weights/ec-v1 --base models/Qwen3.5-0.8B-Base --request request.json --json
```

```json
{"sections": {"门诊": ["..."], "入院": "...", "影像": ["..."], "病理": ["..."], "手术": ["..."]},
 "questions": ["figo", "muscle_invasion"]}
```

The output lists, per question, the chosen option, a confidence in [0, 1] derived from the distribution, and the probabilities over all options (see `fugojev.api`). Add `--device cpu` to run without a GPU.

## Reproduce

With `records.jsonl` (provided separately; it contains patient notes and must not be committed) placed at `data/ec/records.jsonl`:

```bash
python -m fugojev.evaluate --run weights/ec-v1 --base models/Qwen3.5-0.8B-Base --split dev --dtype bf16
```

This should print the fugojev column of the table above. The other columns and the runtime figures:

```bash
python -m fugojev.baseline --split dev --out reports/dev/rule.json
python -m fugojev.base_probe --model models/Qwen3.5-0.8B-Base --split dev --out reports/dev/zero-shot-0.8b.json
python -m fugojev.evaluate --run weights/ec-v1 --base models/Qwen3.5-0.8B-Base --split dev --dtype bf16 \
    --out reports/dev/fugojev-0.8b.json --rows runs/dev-rows.json
python -m fugojev.calibrate --rows runs/dev-rows.json
python -m fugojev.bench_runtime --run weights/ec-v1 --base models/Qwen3.5-0.8B-Base --split dev
python -m fugojev.compare fugojev=reports/dev/fugojev-0.8b.json rule=reports/dev/rule.json zero-shot=reports/dev/zero-shot-0.8b.json
```

To retrain:

```bash
python -m fugojev.train --backbone models/Qwen3.5-0.8B-Base --out runs/ec --epochs 4 --lora 8 --max-state 4096
python -m fugojev.evaluate --run runs/ec/epoch-3 --base models/Qwen3.5-0.8B-Base --split dev --dtype bf16
```

Cross-validation and ablations (about 3 h training plus 25 min scoring on one 4090; `reports/cv/` holds the pooled summary):

```bash
python -m fugojev.cv train --backbone models/Qwen3.5-0.8B-Base --out runs/cv --epochs 3
python -m fugojev.cv score --out runs/cv --epoch 3          # 5 folds × {none, paraphrase, strip-template, pathology-only} × {model, rule}
python -m fugojev.cv report --out runs/cv                   # writes runs/cv/cv-summary.{json,md}
python -m fugojev.evaluate --run weights/ec-v1 --base models/Qwen3.5-0.8B-Base --split dev --ablation paraphrase
```

`data/`, `runs/` and `models/` are git-ignored. Never commit notes, names, ID numbers or admission numbers.

## License

Apache-2.0. Adapted from MedJev (Jun Ma) and kev (Jared Palmer); see `NOTICE`. The backbone is released by Qwen under Apache-2.0.
