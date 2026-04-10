# UNICC AI Safety Evaluation System

A six-layer pipeline that evaluates AI agents for safety, compliance, and governance alignment before admission to the UNICC AI Sandbox. The system accepts text output from any AI agent (in any of 200+ languages), translates it to English via NLLB-200, runs three independent expert evaluations using a local Qwen SLM, and produces a structured council decision (PASS / CONDITIONAL / HOLD / REJECT) backed by a complete audit trail. All inference is local — no external API calls are permitted at any layer.

## System Architecture

The pipeline comprises six independent layers with unidirectional data flow:

| Layer | Name | Responsibility |
|-------|------|----------------|
| L1 | Screening | Input validation, language detection |
| L2 | Multilingual Support | NLLB translation to English, confidence scoring |
| L3 | Expert Council | Three independent Qwen-based expert evaluations (Security, Content, Governance) |
| L4 | Arbitration | Rule-based six-tier decision (no ML) |
| L5 | Resolution | HOLD handling, governance action assembly |
| L6 | Final Output | council_output.json assembly, audit log |

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list

Key dependencies: `transformers`, `torch`, `langdetect`, `sentencepiece`, `fastapi`, `uvicorn`, `pydantic>=2.0`

## Installation

```bash
git clone https://github.com/your-org/unicc-ai-safety-lab.git
cd unicc-ai-safety-lab
pip install -r requirements.txt
```

## Model Setup

Models are downloaded automatically on first run via HuggingFace. Set `HF_TOKEN` in your environment if accessing gated models.

| Model | Purpose |
|-------|---------|
| `facebook/nllb-200-distilled-600M` | L2 translation (200+ languages → English) |
| `Qwen/Qwen2.5-7B-Instruct` | L3 Expert Council + L5 Resolution (production) |
| `Qwen/Qwen2.5-3B-Instruct` | L3 Expert Council + L5 Resolution (local dev, `LOCAL_DEV=true`) |

## Environment Variables

Copy `.env.example` and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | Qwen model to use for experts and resolution |
| `LOCAL_DEV` | `true` | Use 3B model on CPU with float32 when true |
| `UNCERTAINTY_THRESHOLD` | `0.60` | Translation confidence below which input is held for human review |

## Running Locally (CPU, LOCAL_DEV=true)

```bash
LOCAL_DEV=true python main.py --input tests/sample_inputs/verimedia_sample.txt
```

## Running the FastAPI Server

```bash
uvicorn main:app --reload
```

Then POST to `/evaluate`:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "MyAgent", "text": "Agent output text here"}'
```

## Running Tests

```bash
pytest tests/ -v
```

## Known Limitations

This system evaluates text output samples produced by an AI agent or tool; it does not evaluate model weights, training data, or source code, and all conclusions are bounded to the submitted text sample. Non-English inputs are evaluated through NLLB translation to English — translation confidence at or above 0.80 supports a working semantic equivalence assumption, but conclusions for lower-confidence translations carry uncertainty proportional to translation quality and are flagged accordingly. Valid Multilingual Jailbreak scoring requires paired multi-language submissions; single-language submissions default to LOW for that dimension. Multi-modal content (images, audio, video) is outside current scope.
