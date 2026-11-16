"""
commit_prescription_updates.py
Commits each updated and new file individually for the Precision Prescription & VRA Add-on.
"""

import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

commits = [
    ("src/prescription.py", "feat(analytics): implement 4-tier variable rate prescription and economic savings calculator"),
    ("src/database.py", "feat(database): add prescriptions table and logging methods to SQLite store"),
    ("src/predict.py", "fix(inference): enhance dynamic checkpoint resolution for named models"),
    ("main.py", "feat(cli): add prescribe subcommand with customizable field and economic parameters"),
    ("app/app.py", "feat(ui): add dedicated Tab 4 for Variable Rate Application (VRA) Prescription & Economics"),
    ("outputs/reports/cli_prescription_map.png", "results(prescription): add multi-panel VRA prescription and economics visualization"),
    ("outputs/reports/prescription_map_demo.png", "results(prescription): add 4-color VRA prescription grid sample"),
    ("outputs/reports/dji_agras_drone_mission.csv", "results(drone): add autonomous DJI Agras spraying mission waypoint coordinates"),
    ("outputs/agrifusion.db", "feat(database): update agrifusion.db with prescription audit records"),
]

for file_path, msg in commits:
    full_p = os.path.join(PROJECT_ROOT, file_path)
    if os.path.exists(full_p):
        subprocess.run(["git", "add", file_path], cwd=PROJECT_ROOT, check=True)
        diff_res = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT)
        if diff_res.returncode != 0:
            subprocess.run(["git", "commit", "-m", msg], cwd=PROJECT_ROOT, check=True)
            print(f"[COMMITTED] {file_path} -> '{msg}'")
        else:
            print(f"[SKIP] No changes staged for {file_path}")
    else:
        print(f"[WARN] File not found: {file_path}")

