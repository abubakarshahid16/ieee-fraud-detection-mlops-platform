import json
import subprocess
from pathlib import Path


def test_intelligent_trigger_writes_json(tmp_path: Path):
    output_path = tmp_path / "trigger.json"
    subprocess.run(
        [
            "python",
            "-m",
            "fraud_mlops.intelligent_trigger",
            "--event",
            "drift",
            "--metric",
            "fraud_data_drift_score",
            "--value",
            "0.21",
            "--threshold",
            "0.15",
            "--output",
            str(output_path),
        ],
        check=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["action"] == "trigger_retraining"
