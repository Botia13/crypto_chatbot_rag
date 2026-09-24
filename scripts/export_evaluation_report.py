import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from RAG_model.ingestion.config import BASELINE_RUN_CONFIG, SYSTEM_PROMPT


def get_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/summary.json"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.summary_csv)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": get_commit(),
        "system_prompt": SYSTEM_PROMPT,
        "run_config": BASELINE_RUN_CONFIG,
        "summary": frame.where(
            pd.notna(frame),
            None,
        ).to_dict(orient="records"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
    

# uv run python scripts/export_evaluation_report.py `
#     --summary-csv "C:\Users\david\Documents\Projects\crypto_chatbot_rag\data\rag_evaluation\results\v5\2026_09_22_1838\summary.csv"
