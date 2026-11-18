import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PIPELINE = [
    "coverage_and_descriptive.py",
    "regression_analysis.py",
    "contrail_analysis.py",
    "efficiency_analysis.py",
]

for script in PIPELINE:
    path = SCRIPTS_DIR / script
    print(f"\n>>> Running {script} ...")
    res = subprocess.run([sys.executable, str(path)], check=False)
    if res.returncode != 0:
        print(f"[STOP] {script} exited with code {res.returncode}.")
        sys.exit(res.returncode)

print("\n[ALL DONE] All tables saved to reports/tables/ and figures to reports/figures/.")