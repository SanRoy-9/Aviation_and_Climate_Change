import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PIPELINE = [
    "get_eurostat_data.py",
    "get_route_traffic_data.py",
    "get_era5_data.py",
    "coverage_and_descriptive.py",
    "regression_analysis.py",
    "efficiency_analysis.py",
    "contrail_analysis.py",
]

for script in PIPELINE:
    script_name = script if script.endswith(".py") else f"{script}.py"
    path = SCRIPTS_DIR / script_name
    if not path.exists():
        print(f"[ERROR] Missing file: {path}")
        print(f"Create '{script_name}' inside '{SCRIPTS_DIR}' first.")
        sys.exit(1)

    print(f"\n>>> Running {script_name} ...")
    res = subprocess.run([sys.executable, str(path)], check=False)
    if res.returncode != 0:
        print(f"[STOP] {script_name} exited with code {res.returncode}.")
        sys.exit(res.returncode)

print("\n[ALL DONE] All tables saved to reports/tables/ and figures to reports/figures/.")