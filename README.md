# Text generation reference implementation

New implementation added in October 2026. Despite this repository's historical
name, no original course PDF or original source was available for this update.
This is not recovered coursework and does not claim historical results.

## Run without dependencies (Python 3.10+)

```bash
python generate.py --prompt Engineering --max-new-tokens 30
python -m unittest -v
```

The default backend learns word-to-word transition frequencies from
`sample_corpus.txt`, then samples a seeded continuation. It is a small statistical
baseline, not a neural language model. Unknown or terminal words stop generation.
Use `--corpus your_file.txt` for another UTF-8 corpus. The included corpus was
newly authored for this demo; it is not a benchmark dataset.

## Optional actual GPT-Neo inference

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-gpt-neo.txt
python generate.py --backend gpt-neo --prompt "Engineering begins with" --max-new-tokens 40
```

The default is EleutherAI/gpt-neo-125m. First use downloads public model/tokenizer
files from Hugging Face; allow several GB of RAM and sufficient disk space.
CPU is default; `--device cuda` checks CUDA availability. No API key is required.
Weights are not committed. Upstream model licenses and model-card limitations
apply. Model revisions are not pinned; upstream changes may affect reproducibility.
The model is a base text completer, not a reliable instruction-following assistant.
Do not send private documents to untrusted model repositories.

Flags include `--seed`, `--temperature` (neural backend only), `--model` and
`--max-new-tokens` (1-512). Oversized neural prompts are rejected, not truncated.
Output is JSON with separate prompt and continuation fields.

## Validation

Six stdlib tests cover deterministic transitions, unknown tokens, seed
repeatability, length bounds, invalid inputs and actual CLI execution.
GPT-Neo model inference has **not** been executed in the build environment;
PyTorch/Transformers are optional dependencies absent there. No neural output
or performance claim is made. Seed repeatability may differ across devices.

Implementation references:
- https://huggingface.co/docs/transformers/v4.53.0/model_doc/gpt_neo
- https://huggingface.co/EleutherAI/gpt-neo-125m
