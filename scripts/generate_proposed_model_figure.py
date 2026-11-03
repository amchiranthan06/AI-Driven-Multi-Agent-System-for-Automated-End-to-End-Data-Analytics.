from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ink": "#172B4D",
    "muted": "#536579",
    "line": "#8A9AAF",
    "blue": "#2864DC",
    "teal": "#138A83",
    "violet": "#7650C8",
    "orange": "#D45A31",
    "rose": "#C23763",
    "green": "#247A4B",
    "pale_blue": "#EFF5FF",
    "pale_teal": "#EFF9F7",
    "pale_violet": "#F5F1FF",
    "pale_orange": "#FFF4EE",
    "pale_rose": "#FFF1F5",
    "pale_green": "#F0F8F2",
    "panel": "#F6F8FB",
    "white": "#FFFFFF",
}


def rounded_box(ax, xy, width, height, title, body, color, fill, title_size=11.5, body_size=8.4):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.13",
        linewidth=1.7, edgecolor=color, facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height - 0.27, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=COLORS["ink"])
    ax.text(x + width / 2, y + height / 2 - 0.13, body, ha="center", va="center",
            fontsize=body_size, color=COLORS["muted"], linespacing=1.35)


def arrow(ax, start, end, color=None, rad=0.0, lw=1.7, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4,
    ))


fig, ax = plt.subplots(figsize=(15, 7.9), dpi=200)
fig.patch.set_facecolor(COLORS["white"])
ax.set_facecolor(COLORS["white"])
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.85)
ax.axis("off")

# Input and profiling
rounded_box(ax, (0.75, 6.72), 2.55, 0.92, "CSV upload",
            "Upload CSV\nChoose target (optional)", COLORS["blue"], COLORS["pale_blue"], 11, 8.2)
rounded_box(ax, (4.35, 6.72), 4.05, 0.92, "Local storage + chunked profile",
            "CSV saved locally\nProfile read in 100,000-row chunks", COLORS["teal"], COLORS["pale_teal"], 11, 8.2)
rounded_box(ax, (9.45, 6.72), 4.05, 0.92, "Dataset profile",
            "Schema, missing values, numeric summaries,\nand target-column suggestions", COLORS["violet"], COLORS["pale_violet"], 11, 8.0)
arrow(ax, (3.32, 6.84), (4.27, 6.84))
arrow(ax, (8.43, 6.84), (9.37, 6.84))

# Agent cards arranged as a serpentine sequence: 1 -> 2 -> 3, then 4 -> 5 -> 6.
agent_w, agent_h = 3.65, 1.18
left_x, mid_x, right_x = 0.72, 5.68, 10.63
top_y, bottom_y = 4.78, 2.62

rounded_box(ax, (left_x, top_y), agent_w, agent_h,
            "1  Data Preprocessing Agent",
            "Profiles quality; removes duplicate rows within the\nbounded ≤50,000-row modeling sample; retains missing\nvalues for training-pipeline imputation",
            COLORS["teal"], COLORS["pale_teal"], 10.5, 7.1)
rounded_box(ax, (mid_x, top_y), agent_w, agent_h,
            "2  Model Selection and Training Agent",
            "Filters non-predictive columns and preprocesses\nfeatures; compares three task-specific candidates",
            COLORS["blue"], COLORS["pale_blue"], 10.2, 7.9)
rounded_box(ax, (right_x, top_y), agent_w, agent_h,
            "3  Model Evaluation Agent",
            "Reports held-out classification or regression\nmetrics computed by the Python backend",
            COLORS["violet"], COLORS["pale_violet"], 10.5, 8.0)

rounded_box(ax, (right_x, bottom_y), agent_w, agent_h,
            "4  Visualization Agent",
            "Suggests useful charts based on the question\nand previous agent outputs",
            COLORS["orange"], COLORS["pale_orange"], 10.5, 8.0)
rounded_box(ax, (mid_x, bottom_y), agent_w, agent_h,
            "5  Insight Agent",
            "Explains computed patterns, model results, risks,\nand possible next investigations",
            COLORS["rose"], COLORS["pale_rose"], 10.5, 8.0)
rounded_box(ax, (left_x, bottom_y), agent_w, agent_h,
            "6  Report Agent",
            "Combines prior outputs into a concise report\nwith findings and recommended next actions",
            COLORS["green"], COLORS["pale_green"], 10.5, 8.0)

# Main workflow arrows
arrow(ax, (left_x + agent_w + 0.08, top_y + 0.12),
      (mid_x - 0.08, top_y + 0.12))
arrow(ax, (mid_x + agent_w + 0.08, top_y + 0.12),
      (right_x - 0.08, top_y + 0.12))
arrow(ax, (right_x + agent_w / 2, top_y - 0.08),
      (right_x + agent_w / 2, bottom_y + agent_h + 0.08))
arrow(ax, (right_x - 0.08, bottom_y + 0.12),
      (mid_x + agent_w + 0.08, bottom_y + 0.12))
arrow(ax, (mid_x - 0.08, bottom_y + 0.12),
      (left_x + agent_w + 0.08, bottom_y + 0.12))

# Route the profile around the outside edge of the agent cards.
ax.plot([9.55, 9.2, 0.46], [6.68, 6.12, 6.12], color=COLORS["line"], linewidth=1.45)
ax.plot([0.46, 0.46], [6.12, 5.40], color=COLORS["line"], linewidth=1.45)
arrow(ax, (0.46, 5.40), (0.80, 5.40), color=COLORS["line"], lw=1.45)
ax.text(5.1, 6.34, "profile + local dataset reference", ha="center", va="center",
        fontsize=8, color=COLORS["muted"], bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

# Shared state and model boundary
shared = FancyBboxPatch((0.75, 1.8), 13.5, 0.48,
                        boxstyle="round,pad=0.02,rounding_size=0.12",
                        linewidth=1.1, edgecolor="#C8D2DF", facecolor=COLORS["panel"])
ax.add_patch(shared)
ax.text(7.5, 2.04,
        "Shared LangGraph state: question  |  dataset reference  |  profile  |  target  |  model metrics  |  accumulated agent outputs",
        ha="center", va="center", fontsize=8.5, color=COLORS["ink"])

# Final backend artifacts and UI
rounded_box(ax, (2.7, 0.47), 5.5, 0.91, "Backend chart renderer (executes)",
            "Schema-driven Matplotlib charts, stored as local PNGs;\nLLM-assisted fallback with a bounded CSV excerpt when\nschema-driven logic produces no chart",
            COLORS["orange"], COLORS["pale_orange"], 10.5, 7.1)
rounded_box(ax, (9.2, 0.47), 4.8, 0.91, "Final frontend output",
            "Report, metrics, agent statuses, and chart artifacts", COLORS["green"], COLORS["pale_green"], 10.5, 8.1)
arrow(ax, (2.55, bottom_y - 0.08), (2.55, 2.30), color=COLORS["line"], lw=1.45)
arrow(ax, (5.45, 1.78), (5.45, 1.42), color=COLORS["line"], lw=1.45)
arrow(ax, (8.29, 0.92), (9.12, 0.92))

# Execution notes
ax.text(7.5, 0.19,
        "Python/Pandas/scikit-learn compute profiles and metrics  •  Local Ollama/Qwen3-4B supports selected explanations  •  Agent fallback statuses are surfaced",
        ha="center", va="center", fontsize=8.1, color=COLORS["muted"])

fig.savefig(OUTPUT_DIR / "proposed_model_workflow.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_DIR / "proposed_model_workflow.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUTPUT_DIR / "proposed_model_workflow.png")
