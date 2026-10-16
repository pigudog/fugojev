# fugojev

**Fu**dan University **G**ynecologic **O**ncology **Jev** — a [MedJev](https://github.com/JunMa11/MedJev)-style decision model built specifically for gynecologic oncology.

The goal is fast FIGO staging of gynecologic cancers from clinical notes on a CPU or a small GPU: a sub-1B frozen backbone, a 42 MB adapter, and closed-set answers with probabilities instead of generated text. **Version 1.1.0** (tag `v1.1.0`) solves endometrial cancer, FIGO 2009; cervical and ovarian cancer are planned for later versions. 1.1 adds the reference systems (rule baseline, zero-shot probe), calibration, a runtime benchmark and a JSON request API on top of the 1.0 model, whose weights are unchanged.

Given the pathology, imaging, outpatient, admission and operative notes of one patient, the model answers five multiple-choice questions: FIGO stage, depth of myometrial invasion, lymphovascular space invasion, lymph-node extent, and histologic grade.

## How it works

A frozen `Qwen/Qwen3.5-0.8B-Base` backbone with a LoRA adapter (r=8, 5.9M trainable parameters) encodes the notes once. Each question is appended as a separate branch that can see the notes but not the other questions, and a pointer head scores the option tokens; softmax over the allowed options gives the answer distribution. If the myometrial-invasion head says ≥1/2, a FIGO prediction of IA is lifted to IB. Training: 4 epochs, AdamW 1e-4, bf16 autocast, about 50 minutes on one RTX 4090.

## Results

321 patients with notes, split by hashed admission number into train 257 / dev 36 / test 28. Dev set (36 patients, 169 questions; used for model selection, so not a final score — the test set has not been opened). Accuracy (macro-F1); `python -m fugojev.compare` produces this table from the reports in `reports/dev/`:

| Question | fugojev 0.8B | rule baseline | zero-shot 0.8B | fugojev 0.5B | majority |
|---|---|---|---|---|---|
| FIGO stage (9, n=36) | **0.972** (0.982) | 0.917 (0.905) | 0.083 (0.062) | 0.750 (0.318) | 0.611 |
| Myometrial invasion (3, n=34) | **1.000** (1.000) | 1.000 (1.000) | 0.382 (0.243) | 0.912 (0.837) | 0.618 |
| Lymph-node extent (4, n=36) | 0.972 (0.929) | **1.000** (1.000) | 0.111 (0.129) | 0.889 (0.471) | 0.889 |
| LVSI (2, n=31) | **1.000** (1.000) | 1.000 (1.000) | 0.806 (0.446) | 0.871 (0.713) | 0.839 |
| Grade (3, n=32) | **1.000** (1.000) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 0.500 |
| micro (169 questions) | **0.988** | 0.982 | 0.456 | 0.882 | |
| latency per patient (4090) | 233 ms | — | 416 ms | 89 ms | |

- *rule baseline* (`fugojev.baseline`): regular expressions over the templated pathology fields (`肌层浸润:`, `脉管侵犯:（-）`, `淋巴结转移情况:总数（0/24）` …), composed into a stage; patterns tuned on train only. On this centre's templated reports it is a strong floor: the components are read off exactly, and the gap to the model is in stage composition (train FIGO 0.879 vs 0.91). The model's advantage is that it was not written for this template.
- *zero-shot* (`fugojev.base_probe`): the same base model, no adapter, answer read from next-token letter logits. Below the majority floor on every question except grade, so the score comes from the training, not the backbone alone.
- *fugojev 0.5B*: same recipe on Qwen2.5-0.5B.

Calibration (`fugojev.calibrate`): the raw probabilities are already calibrated on dev (FIGO ECE 0.078, NLL 0.183; every other question ECE < 0.02). Leave-one-out temperature + prior fitting does not improve FIGO NLL on 36 patients, so no calibration file is shipped; the tool is there for a larger set.

Runtime (`fugojev.bench_runtime`): on an RTX 4090 in bf16, reading the notes costs 228 ms and each additional question about 1.5 ms (one question 229 ms, all five 234 ms, p95 243 ms); latency is flat from 1k to 3k state tokens; peak memory 1.9 GB. On CPU (fp32, 80 cores) about 6 s per patient with the same answers.

## Install

```bash
conda create -n fugojev python=3.12 -y && conda activate fugojev
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev,qwen35]" --upgrade-strategy only-if-needed
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

`data/`, `runs/` and `models/` are git-ignored. Never commit notes, names, ID numbers or admission numbers.

## License

Apache-2.0. Adapted from MedJev (Jun Ma) and kev (Jared Palmer); see `NOTICE`. The backbone is released by Qwen under Apache-2.0.
