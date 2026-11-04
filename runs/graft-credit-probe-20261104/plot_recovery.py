"""Render the recorded recovery observations; requires optional matplotlib."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = json.loads((root / "curves.json").read_text())["rows"]
seeds = sorted({row["seed"] for row in rows})
styles = {
    "sham": ("Sham + adaptation", "#64748b", "--"),
    "disrupted_rewire": ("Support replacement + adaptation", "#15803d", "-"),
    "disrupted_no_rewire": ("Support replacement + fixed support", "#c2410c", "-"),
}
fig, axes = plt.subplots(1, len(seeds), figsize=(12, 4), sharey=True, squeeze=False)
for ax, seed in zip(axes[0], seeds):
    for branch, (label, color, linestyle) in styles.items():
        points = sorted((row for row in rows if row["seed"] == seed and row["phase"] == "recovery"
                         and row["branch"] == branch), key=lambda row: row["step"])
        ax.plot([p["step"] for p in points], [p["mse"] for p in points], label=label,
                color=color, linestyle=linestyle, linewidth=1.8)
    ax.set_title(f"Seed {seed}")
    ax.set_xlabel("Recovery update")
    ax.grid(alpha=.2)
max_error = max(row["mse"] for row in rows if row["phase"] == "recovery")
axes[0, 0].set_ylim(0, max_error * 1.05)
axes[0, 0].set_ylabel("Current mean squared error")
fig.suptitle("Supervised graft-current recovery after a declared target switch", fontsize=13)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3,
           bbox_to_anchor=(.5, .035), frameon=False, fontsize=9)
fig.text(.5, .015, "Fixed 12/24 active records; equal recovery data. Step 0 is after the switch and support intervention.",
         ha="center", fontsize=9, color="#475569")
fig.tight_layout(rect=(0, .14, 1, .94))
fig.savefig(root / "recovery.png", dpi=160)
fig.savefig(root / "recovery.svg")
print(root / "recovery.png")
