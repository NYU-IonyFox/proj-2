"""
main.py
FastAPI app (GET /health, POST /evaluate) + CLI entry point.

FastAPI and CLI share the same run_council() function — no duplicate logic.

FAIL-CLOSED: any unhandled exception in POST /evaluate returns HTTP 200
with HOLD(uncertainty) response.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

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
    agent_name: str = ""
    text: str = None
    repo_url: str = None


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
        return run_council(
            agent_name=req.agent_name,
            text=req.text,
            repo_url=req.repo_url,
        )
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
        help="Path to a text file containing the AI agent output to evaluate.",
    )
    parser.add_argument(
        "--repo",
        help="GitHub repository URL to evaluate (alternative to --input)",
    )
    args = parser.parse_args()

    from output.final_output import run_council

    if args.repo:
        result = run_council(repo_url=args.repo)
    elif args.input:
        from pathlib import Path
        p = Path(args.input)
        agent_name = p.stem
        text = p.read_text(encoding="utf-8")
        result = run_council(agent_name, text)
    else:
        print("Error: provide --input <file> or --repo <github_url>", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if "--input" in sys.argv or "--repo" in sys.argv:
        _cli()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
