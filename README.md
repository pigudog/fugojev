# fugojev

**Fu**dan University **G**ynecologic **O**ncology **Jev** — a [MedJev](https://github.com/JunMa11/MedJev)-style decision model built specifically for gynecologic oncology.

The goal is fast FIGO staging of gynecologic cancers from clinical notes on a CPU or a small GPU: a sub-1B frozen backbone, a 42 MB adapter, and closed-set answers with probabilities instead of generated text. **Version 1.3.0** (tag `v1.3.0`) solves endometrial cancer in both FIGO 2009 and FIGO 2023; cervical and ovarian cancer are planned for later versions. 1.1 added the reference systems (rule baseline, zero-shot probe), calibration, a runtime benchmark and a JSON request API; 1.2 added five-fold cross-validation and input ablations, which revised the headline: on this centre's templated pathology the rule baseline stages as well as or better than the model, and the model's advantage appears only when the template is removed. 1.3 adds the FIGO 2023 stage, the inputs it needs (immunohistochemistry and molecular pathology reports) and seven new questions, and ships new weights (`weights/ec-v1.3`); the 1.0 weights stay in `weights/ec-v1`.

Given the pathology, immunohistochemistry, molecular, imaging, outpatient, admission and operative notes of one patient, the model answers twelve multiple-choice questions: FIGO 2009 stage, FIGO 2023 stage, histotype, depth of myometrial invasion, lymphovascular space invasion, cervical stromal invasion, highest extrauterine site, lymph-node extent, histologic grade, p53 status, mismatch-repair status and POLE status. The FIGO 2023 stage is also composed from the component answers by the staging rule, which is the recommended output.

## How it works

A frozen `Qwen/Qwen3.5-0.8B-Base` backbone with a LoRA adapter (r=8, 5.9M trainable parameters) encodes the notes once. Each question is appended as a separate branch that can see the notes but not the other questions, and a pointer head scores the option tokens; softmax over the allowed options gives the answer distribution. If the myometrial-invasion head says ≥1/2, a FIGO 2009 prediction of IA is lifted to IB. The FIGO 2023 stage is composed from the ten component answers by `fugojev.figo2023.compose`, the same function that built the gold labels; the direct `figo2023` head is reported for comparison. Training: 4 epochs, AdamW 1e-4, bf16 autocast; with twelve questions about 2.7 hours on one RTX 4090 (the 1.0 five-question model took 50 minutes).

## Results

321 patients with notes, split by hashed admission number into train 257 / dev 36 / test 28. The test set has not been opened.

### FIGO 2023 (v1.3)

**Gold labels.** The case sheet records the FIGO 2009 stage, the components (histotype, grade, myometrial depth, LVSI, node stations) and the research molecular class (POLEmut 47, MMRd 79, p53abn 57, NSMP 148 of 333). The 2023 stage is derived by `fugojev.figo2023.compose` from these columns plus template cells of the pathology report (cervical stroma, ovaries and tubes, parametria, vagina) and a per-patient review of the 2009 stage-IV cases (operative and pathology notes decide IVB abdominal peritoneum vs IVC inguinal or retrocrural nodes; the review is a small override file that is not distributed). When the sheet disagrees with its own components (three cases, e.g. IA with a positive pelvic node) the components win. Conventions fixed with the clinicians before training: *aggressive* histology is non-endometrioid or endometrioid G3; this centre's reports give LVSI as present/absent without a vessel count, so any LVSI in a non-aggressive uterus-confined tumour is staged IIB; node micro/macro-metastasis is not reported, so IIIC1/IIIC2 carry no i/ii suffix; POLEmut in stage I–II gives IAmPOLEmut and p53abn with myometrial invasion gives IICmp53abn, MMRd and NSMP leave the stage unchanged. Distribution over the 301 labelled patients: IA2 86, IAmPOLEmut 45, IICmp53abn 33, IA1 24, IB 23, IIIC1 20, IIB 18, IIC 16, IIIC2 11, IVB 10, IIA 8, IIIA1 8, IVC 4, IC 3, IIIB1 2, IIIB2 2 (20 patients lack an LVSI or depth entry and get no 2023 label).

**Inputs.** The notes of 302/321 patients contain a p53 immunostain, 288 a mismatch-repair panel and 136 a sequencing report; version 1 discarded all three as generic laboratory reports. 1.3 renders them as `[免疫组化]` and `[分子病理]` sections. Molecular reports are 2.6k characters with the findings at the top and method boilerplate at the bottom, so unlike the other sections they are condensed to the result lines that mention POLE, TP53 or the MMR genes rather than tail-truncated. The POLE question is only asked when a molecular report is present (143 patients, 19 positive). The research molecular class itself is a gold column and never enters the notes.

**Three ways to the 2023 stage.** `figo2023` is the direct 19-way head. `figo2023_composed` applies the staging rule to the ten component heads, reading the molecular class from the notes (p53 and MMR from immunohistochemistry, POLE from the sequencing report when there is one). `figo2023_composed_given` is the same with the molecular class supplied as structured input, as a clinic holding an outside sequencing report would do (`--molecular POLEmut` in `predict`). The notes alone cannot reach the gold label for every patient. Among the 182 non-test patients with every component labelled, 15 whose gold stage is IAmPOLEmut have no sequencing report in their notes, and in 19 the immunostain and the research p53 call disagree (the research class used sequencing). With every component read perfectly, notes-only composition therefore tops out at 148/182 = 0.81; with the molecular class given it reaches 1.00.

Development set (36 patients, 30 with a 2023 label), `weights/ec-v1.3` (epoch 2 of 4, chosen on dev) against the rule baseline; accuracy (macro-F1):

| Question | fugojev 0.8B | rule baseline | majority |
|---|---|---|---|
| FIGO 2023, composed, molecular given (n=30) | **1.000** (1.000) | 0.933 (0.981) | 0.233 |
| FIGO 2023, composed from notes (n=30) | **0.767** (0.777) | 0.667 (0.696) | 0.233 |
| FIGO 2023, direct head (n=30) | 0.633 (0.528) | — | 0.233 |
| FIGO 2009 (n=36) | 0.917 (0.873) | 0.917 (0.905) | 0.611 |
| Histotype (n=36) | 0.944 (0.662) | 0.944 (0.662) | 0.944 |
| Cervical stroma (n=36) | 1.000 | 1.000 | 0.861 |
| Extrauterine site (n=36) | **1.000** | 0.944 | 0.972 |
| p53 (n=28) | 0.893 (0.801) | 0.893 (0.801) | 0.821 |
| MMR (n=29) | 0.966 (0.958) | 0.966 (0.958) | 0.690 |
| POLE (n=13) | 1.000 | 1.000 | 1.000 |
| Myometrial invasion / LVSI / grade | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | |
| Lymph-node extent (n=36) | 0.972 | **1.000** | 0.889 |
| micro (377 questions) | 0.944 | 0.944 | |

The direct head is the weakest route: with 19 classes and 245 training labels it defaults to IA2 for the molecular subtypes. Composition is better because the component heads are at or near ceiling and the rule carries the rare stages. The seven composed-from-notes misses are the cases described above (four patients without a sequencing report whose gold is IAmPOLEmut, two where the immunostain and the research p53 call disagree, one p53 misread), which is why the molecular-given row is the one to quote for a clinic with a classifier result in hand. Later epochs raised the direct head (0.700 at epoch 4) but began to call extrauterine spread from imaging mentions that pathology did not confirm, so epoch 2 is shipped. Five-fold cross-validation of the 1.3 recipe is in `reports/cv/cv-summary-v1.3.md` and summarised below.

**Five-fold cross-validation (293 patients, 275 with a 2023 label).** Same folds and protocol as the 2009 table below, one twelve-question model per fold, 3 epochs fixed in advance. Pooled accuracy, 95% patient bootstrap CI on the difference, exact McNemar p:

| Question | fugojev 0.8B | rule baseline | Δ [95% CI] | McNemar p | majority |
|---|---|---|---|---|---|
| FIGO 2023, composed, molecular given (n=275) | 0.876 [0.837, 0.915] | **0.895** [0.857, 0.930] | −0.018 [−0.061, 0.022] | 0.49 | 0.276 |
| FIGO 2023, composed from notes (n=275) | 0.658 | 0.669 | −0.011 [−0.061, 0.039] | 0.78 | 0.276 |
| FIGO 2023, direct head (n=275) | 0.560 | — | −0.109 vs rule composed | 0.005 | 0.276 |
| FIGO 2009 (n=292) | 0.863 | 0.887 | −0.024 [−0.058, 0.010] | 0.23 | 0.630 |
| Histotype (n=293) | 0.918 | 0.945 | −0.027 [−0.058, 0.003] | 0.13 | 0.870 |
| Cervical stroma (n=290) | 0.983 | 0.979 | +0.003 | 1.00 | 0.879 |
| Extrauterine site (n=293) | 0.898 | 0.932 | −0.034 [−0.068, 0.000] | 0.09 | 0.911 |
| p53 (n=237) | 0.899 | 0.932 | −0.034 [−0.071, 0.004] | 0.13 | 0.793 |
| MMR (n=258) | 0.969 | 0.973 | −0.004 | 1.00 | 0.752 |
| POLE (n=132) | 0.947 | 0.962 | −0.015 | 0.50 | 0.871 |
| Grade (n=267) | **0.989** | 0.963 | +0.026 [0.004, 0.052] | 0.07 | 0.423 |
| Myometrial invasion / LVSI / lymph nodes | 0.990 / 0.996 / 0.965 | 0.993 / 0.993 / 0.979 | | | |
| micro (twelve heads) | 0.913 | 0.933 | | | |

Per fold, composed-with-molecular-given is 0.94 / 0.79 / 0.93 / 0.84 / 0.87 for the model and 0.96 / 0.79 / 0.90 / 0.92 / 0.91 for the rules, so the dev row (1.000 on 30 patients) is again at the lucky end. The pattern of 1.2 repeats: the two systems are statistically indistinguishable on every component and on the composed stage, the direct 19-way head is clearly worse than either, and the model's remaining losses sit in one place: of its 34 misses with the molecular class given, 17 are an extrauterine-site error alone (mostly IIIC2 called IVC, i.e. para-aortic nodes read as distant, and IICmp53abn called IVB from peritoneal mentions), another 8 involve the site together with another component. Under `paraphrase` (`reports/cv/cv-summary-v1.3.md`) the 1.2 picture holds and extends to the stage: the rules fall to 0.736 on myometrial invasion and 0.871 on LVSI while the model stays at 0.986 / 0.993, and the composed-from-notes stage is now better for the model, 0.545 vs 0.465 (+0.080 [0.007, 0.153], p = 0.03); the immunohistochemistry and molecular questions are unchanged because the paraphrase does not touch those sections.

### FIGO 2009: five-fold cross-validation (293 patients, v1.2 models)

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

### FIGO 2009: input ablations (same folds, same v1.2 models)

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

### FIGO 2009: development set (36 patients, v1.0 weights)

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

The repository ships only the LoRA adapters and pointer heads (`weights/ec-v1.3` for the twelve-question 1.3 model, `weights/ec-v1` for the five-question 1.0 model; 42 MB each). Download the backbone (about 1.7 GB) into `models/`:

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3.5-0.8B-Base --local-dir models/Qwen3.5-0.8B-Base
```

Older `huggingface_hub` versions use `huggingface-cli download` with the same arguments. Behind a restricted network, set `HF_ENDPOINT=https://hf-mirror.com` first, or use ModelScope: `modelscope download --model Qwen/Qwen3.5-0.8B-Base --local_dir models/Qwen3.5-0.8B-Base`.

## Predict

```bash
python -m fugojev.predict --run weights/ec-v1.3 --base models/Qwen3.5-0.8B-Base --state note.txt
```

`note.txt` holds the sections in order, `[门诊] [入院] [影像] [病理] [免疫组化] [分子病理] [手术]`, each with its conclusion at the end (the molecular report with its result lines; sections may be missing). Or pass the sections as JSON and let the package render them exactly as the training data was rendered:

```bash
python -m fugojev.predict --run weights/ec-v1.3 --base models/Qwen3.5-0.8B-Base --request request.json --json --molecular p53abn
```

```json
{"sections": {"门诊": ["..."], "入院": "...", "影像": ["..."], "病理": ["..."], "免疫组化": ["..."], "分子病理": ["..."], "手术": ["..."]},
 "molecular": "p53abn",
 "questions": ["figo", "figo2023", "muscle_invasion"]}
```

The output lists, per question, the chosen option, a confidence in [0, 1] derived from the distribution, and the probabilities over all options (see `fugojev.api`). When all ten component questions are asked (the default), a derived `figo2023_composed` answer is added: the FIGO 2023 stage composed from the component answers, with the molecular class taken from the request's `molecular` field (`POLEmut`, `MMRd`, `p53abn`, `NSMP`) when present and otherwise read from the notes; `molecular_source` says which. Add `--device cpu` to run without a GPU.

## Reproduce

With `records.jsonl` (provided separately; it contains patient notes and must not be committed) placed at `data/ec/records.jsonl`:

```bash
python -m fugojev.evaluate --run weights/ec-v1.3 --base models/Qwen3.5-0.8B-Base --split dev --dtype bf16
python -m fugojev.baseline --split dev --out reports/dev/rule-v1.3.json
```

This should print the fugojev and rule columns of the 1.3 development table (`reports/dev/fugojev-0.8b-v1.3.json`, `reports/dev/rule-v1.3.json`). The 2009 development table uses the 1.0 weights; its other columns and the runtime figures:

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
python -m fugojev.evaluate --run runs/ec/epoch-2 --base models/Qwen3.5-0.8B-Base --split dev --dtype bf16
```

`records.jsonl` is built from the raw workbooks with `python -m fugojev build`; the FIGO 2023 labels additionally read `data/ec/gold2023-overrides.json` (path in `FUGOJEV_OVERRIDES`), a JSON object mapping admission number to `{"extent": ...}` for the stage-IV cases reviewed by hand.

Cross-validation and ablations (v1.2 five-question models: about 3 h training plus 25 min scoring on one 4090; the twelve-question 1.3 recipe takes about 9 h; `reports/cv/` holds both pooled summaries):

```bash
python -m fugojev.cv train --backbone models/Qwen3.5-0.8B-Base --out runs/cv --epochs 3
python -m fugojev.cv score --out runs/cv --epoch 3          # 5 folds × {none, paraphrase, strip-template, pathology-only} × {model, rule}
python -m fugojev.cv report --out runs/cv                   # writes runs/cv/cv-summary.{json,md}
python -m fugojev.evaluate --run weights/ec-v1 --base models/Qwen3.5-0.8B-Base --split dev --ablation paraphrase
```

`data/`, `runs/` and `models/` are git-ignored. Never commit notes, names, ID numbers or admission numbers.

## License

Apache-2.0. Adapted from MedJev (Jun Ma) and kev (Jared Palmer); see `NOTICE`. The backbone is released by Qwen under Apache-2.0.
