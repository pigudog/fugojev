# fugojev

**Fu**dan University **G**ynecologic **O**ncology **Jev** — a [MedJev](https://github.com/JunMa11/MedJev)-style decision model built specifically for gynecologic oncology.

The goal is fast FIGO staging of gynecologic cancers from clinical notes on a CPU or a small GPU: a sub-1B frozen backbone, a 42 MB adapter, and closed-set answers with probabilities instead of generated text. **Version 1.0.0** (tag `v1.0.0`) solves endometrial cancer, FIGO 2009; cervical and ovarian cancer are planned for later versions.

Given the pathology, imaging, outpatient, admission and operative notes of one patient, the model answers five multiple-choice questions: FIGO stage, depth of myometrial invasion, lymphovascular space invasion, lymph-node extent, and histologic grade.

## How it works

A frozen `Qwen/Qwen3.5-0.8B-Base` backbone with a LoRA adapter (r=8, 5.9M trainable parameters) encodes the notes once. Each question is appended as a separate branch that can see the notes but not the other questions, and a pointer head scores the option tokens; softmax over the allowed options gives the answer distribution. If the myometrial-invasion head says ≥1/2, a FIGO prediction of IA is lifted to IB. Training: 4 epochs, AdamW 1e-4, bf16 autocast, about 50 minutes on one RTX 4090.

## Results

321 patients with notes, split by hashed admission number into train 257 / dev 36 / test 28. Dev set (36 patients, 169 questions; used for model selection, so not a final score — the test set has not been opened):

| Question | Accuracy | Macro-F1 | Majority baseline |
|---|---|---|---|
| FIGO stage (9 classes) | 0.972 | 0.982 | 0.611 |
| Myometrial invasion (3) | 1.000 | 1.000 | 0.618 |
| Lymph-node extent (4) | 0.972 | 0.929 | 0.889 |
| LVSI (2) | 1.000 | 1.000 | 0.839 |
| Grade (3) | 1.000 | 1.000 | 0.500 |
| All 169 questions | 0.988 | | |

Inference: 239 ms per patient and 1.9 GB GPU memory on an RTX 4090 (bf16); about 6 s per patient on CPU (`--device cpu`, fp32, same dev accuracy within one flip). The same recipe on Qwen2.5-0.5B reaches only 0.750 FIGO accuracy.

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

`note.txt` holds the five sections in order, `[门诊] [入院] [影像] [病理] [手术]`, each with its conclusion at the end. The output lists the chosen option and the probabilities for every question. Add `--device cpu` to run without a GPU.

## Reproduce

With `records.jsonl` (provided separately; it contains patient notes and must not be committed) placed at `data/ec/records.jsonl`:

```bash
python -m fugojev.evaluate --run weights/ec-v1 --base models/Qwen3.5-0.8B-Base --split dev --dtype bf16
```

This should print the numbers in the table above. To retrain:

```bash
python -m fugojev.train --backbone models/Qwen3.5-0.8B-Base --out runs/ec --epochs 4 --lora 8 --max-state 4096
python -m fugojev.evaluate --run runs/ec/epoch-3 --base models/Qwen3.5-0.8B-Base --split dev --dtype bf16
pytest
```

`data/`, `runs/` and `models/` are git-ignored. Never commit notes, names, ID numbers or admission numbers.

## License

Apache-2.0. Adapted from MedJev (Jun Ma) and kev (Jared Palmer); see `NOTICE`. The backbone is released by Qwen under Apache-2.0.
