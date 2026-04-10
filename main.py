"""
main.py
FastAPI app (GET /health, POST /evaluate) + CLI entry point.

FastAPI and CLI share the same run_council() function — no duplicate logic.

FAIL-CLOSED: any unhandled exception in POST /evaluate returns HTTP 200
with HOLD(uncertainty) response.
"""
from __future__ import annotations

import argparse
import json
import sys

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="UNICC AI Safety Lab", version="1.0")


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    agent_name: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=20000)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    import os
    model_id = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    return {"status": "ok", "model_id": model_id, "schema_version": "1.0"}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    try:
        from output.final_output import run_council
        return run_council(req.agent_name, req.text)
    except Exception:  # noqa: BLE001
        return {
            "final_decision": "HOLD",
            "hold_reason": "uncertainty",
            "decision_rule_triggered": "pipeline_error",
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="UNICC AI Safety Lab — evaluate an AI agent output file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a text file containing the AI agent output to evaluate.",
    )
    args = parser.parse_args()

    input_path = args.input
    from pathlib import Path
    p = Path(input_path)
    agent_name = p.stem
    text = p.read_text(encoding="utf-8")

    from output.final_output import run_council
    result = run_council(agent_name, text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
