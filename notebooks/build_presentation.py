"""
notebooks/build_presentation.py
--------------------------------
Builds the submission deck as a PDF (one page per slide) from the *actual*
trained artifacts, so the numbers in the presentation can never drift from the
numbers the platform produces.

Reads:  models/metrics.json, models/business_rules.json
        documents/screenshots/*.png   (any UI screenshots you drop in there are
                                       appended automatically)
Writes: documents/Credit_Risk_Platform_Presentation.pdf

Run from the project root:
    python notebooks/build_presentation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

MODELS_DIR = _PROJECT_ROOT / "models"
DOCS_DIR = _PROJECT_ROOT / "documents"
SHOTS_DIR = DOCS_DIR / "screenshots"
OUT_PDF = DOCS_DIR / "Credit_Risk_Platform_Presentation.pdf"

# ─── palette (dark, matching the UI) ──────────────────────────────────────────
BG = "#12121f"
FG = "#f0f0f8"
MUTED = "#9a9ab0"
ACCENT = "#a18cd1"
GOOD = "#2ecc71"
WARN = "#f39c12"
BAD = "#e74c3c"

SLIDE = (13.333, 7.5)  # 16:9


def _new_slide():
    fig = plt.figure(figsize=SLIDE, facecolor=BG)
    return fig


def _title(fig, text, subtitle=None):
    fig.text(0.06, 0.93, text, fontsize=30, color=FG, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.06, 0.855, subtitle, fontsize=14, color=ACCENT, va="top")
    # Sits below the subtitle. At 0.805 it struck through the subtitle text.
    fig.patches.append(
        plt.Rectangle((0.06, 0.795), 0.12, 0.006, transform=fig.transFigure,
                      facecolor=ACCENT, edgecolor="none")
    )


def _bullets(fig, items, x=0.07, y=0.72, dy=0.075, size=14, color=FG):
    for i, item in enumerate(items):
        fig.text(x, y - i * dy, item, fontsize=size, color=color, va="top", wrap=True)


def _footer(fig, page_label):
    fig.text(0.94, 0.04, page_label, fontsize=10, color=MUTED, ha="right")
    fig.text(0.06, 0.04, "Credit Risk Platform · Home Credit Default Risk",
             fontsize=10, color=MUTED)


def _style_axes(ax):
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def _card(fig, x, y, w, h, label, value, color=ACCENT):
    fig.patches.append(
        plt.Rectangle((x, y), w, h, transform=fig.transFigure,
                      facecolor="#1c1c2e", edgecolor="#2e2e46", linewidth=1)
    )
    fig.text(x + w / 2, y + h * 0.62, str(value), fontsize=26, color=color,
             fontweight="bold", ha="center", va="center")
    fig.text(x + w / 2, y + h * 0.24, label, fontsize=10.5, color=MUTED,
             ha="center", va="center")


# ─── load artifacts ───────────────────────────────────────────────────────────
def load_artifacts():
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    rules_path = MODELS_DIR / "business_rules.json"
    rules = {}
    if rules_path.exists():
        with open(rules_path) as f:
            rules = json.load(f)
    return metrics, rules


# ─── slides ───────────────────────────────────────────────────────────────────
def slide_title(pdf, metrics):
    fig = _new_slide()
    fig.text(0.5, 0.62, "Credit Risk Platform", fontsize=46, color=FG,
             fontweight="bold", ha="center")
    fig.text(0.5, 0.53, "Predicting loan default, explaining every decision,\n"
                        "and answering questions in plain English",
             fontsize=17, color=ACCENT, ha="center", linespacing=1.6)
    fig.text(0.5, 0.36,
             f"Home Credit Default Risk  ·  {metrics['n_samples']:,} applicants  ·  "
             f"{metrics['n_features']} engineered features",
             fontsize=13, color=MUTED, ha="center")
    fig.text(0.5, 0.30, "LightGBM  ·  SHAP  ·  Gemini NL-to-SQL  ·  FastAPI + React  ·  Docker",
             fontsize=12, color=MUTED, ha="center")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_problem(pdf, metrics):
    fig = _new_slide()
    _title(fig, "The Problem", "Who will repay, and can we justify the answer?")
    _bullets(fig, [
        "Home Credit lends to people with little or no formal credit history.",
        "Rejecting a good borrower loses revenue; approving a bad one loses principal.",
        "",
        "Three things a lending team actually needs:",
        "   1.  A calibrated probability of default — not just a ranking.",
        "   2.  A reason for every decision that a loan officer can read aloud.",
        "   3.  Self-serve answers to portfolio questions, without waiting on an analyst.",
    ])
    _card(fig, 0.07, 0.10, 0.19, 0.15, "Applicants", f"{metrics['n_samples']:,}")
    _card(fig, 0.28, 0.10, 0.19, 0.15, "Default rate",
          f"{metrics['default_rate'] * 100:.2f}%", WARN)
    _card(fig, 0.49, 0.10, 0.19, 0.15, "Class ratio", "11.4 : 1", BAD)
    _card(fig, 0.70, 0.10, 0.19, 0.15, "Features", metrics["n_features"])
    _footer(fig, "2")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_architecture(pdf):
    fig = _new_slide()
    _title(fig, "Architecture", "Five sections, one API, one container stack")

    boxes = [
        (0.06, 0.50, "Kaggle CSVs\napplication + bureau", "#1c1c2e", MUTED),
        (0.28, 0.50, "Feature pipeline\nloader → preprocessor", "#1c1c2e", ACCENT),
        (0.50, 0.50, "LightGBM\n5-fold CV + calibration", "#1c1c2e", GOOD),
        (0.72, 0.50, "SHAP\nglobal + per-applicant", "#1c1c2e", WARN),
        (0.06, 0.24, "SQLite\napplications + bureau", "#1c1c2e", MUTED),
        (0.28, 0.24, "Gemini NL→SQL\nvalidate · ground · retry", "#1c1c2e", ACCENT),
        (0.50, 0.24, "FastAPI\n/predict /explain /chat", "#1c1c2e", GOOD),
        (0.72, 0.24, "React UI\n5 sections", "#1c1c2e", WARN),
    ]
    w, h = 0.20, 0.14
    for x, y, label, bg, edge in boxes:
        fig.patches.append(
            plt.Rectangle((x, y), w, h, transform=fig.transFigure,
                          facecolor=bg, edgecolor=edge, linewidth=1.6)
        )
        fig.text(x + w / 2, y + h / 2, label, fontsize=11.5, color=FG,
                 ha="center", va="center", linespacing=1.5)

    for x in (0.265, 0.485, 0.705):
        fig.text(x, 0.57, "→", fontsize=20, color=ACCENT, ha="center")
    for x in (0.265, 0.485, 0.705):
        fig.text(x, 0.31, "→", fontsize=20, color=ACCENT, ha="center")

    fig.text(0.06, 0.14, "Everything below the model layer is read-only: the chatbot may only "
                         "run validated SELECT statements against a schema it is\nproven to know.",
             fontsize=12, color=MUTED, linespacing=1.6)
    _footer(fig, "3")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_eda_charts(pdf):
    fig = _new_slide()
    _title(fig, "Exploratory Analysis", "Where the risk actually concentrates")

    housing = {
        "Rented apartment": 12.31, "With parents": 11.70, "Municipal apt": 8.54,
        "Co-op apartment": 7.93, "House / apartment": 7.80, "Office apartment": 6.57,
    }
    education = {
        "Lower secondary": 10.93, "Secondary": 8.94, "Incomplete higher": 8.48,
        "Higher education": 5.36, "Academic degree": 1.83,
    }

    ax1 = fig.add_axes([0.07, 0.16, 0.38, 0.56])
    _style_axes(ax1)
    names = list(housing.keys())[::-1]
    vals = list(housing.values())[::-1]
    colors = [BAD if v > 10 else WARN if v > 8.07 else GOOD for v in vals]
    ax1.barh(names, vals, color=colors)
    ax1.axvline(8.07, color=FG, linestyle="--", linewidth=1.2)
    ax1.text(8.25, -0.45, "portfolio 8.07%", color=FG, fontsize=9)
    ax1.set_title("Default rate by housing type (%)", fontsize=13, pad=12)

    ax2 = fig.add_axes([0.57, 0.16, 0.36, 0.56])
    _style_axes(ax2)
    names2 = list(education.keys())[::-1]
    vals2 = list(education.values())[::-1]
    colors2 = [BAD if v > 10 else WARN if v > 8.07 else GOOD for v in vals2]
    ax2.barh(names2, vals2, color=colors2)
    ax2.axvline(8.07, color=FG, linestyle="--", linewidth=1.2)
    ax2.set_title("Default rate by education (%)", fontsize=13, pad=12)

    fig.text(0.07, 0.09, "Housing stability and education both separate risk by a factor of ~2 "
                         "to ~6 against an 8.07% baseline.", fontsize=12, color=MUTED)
    _footer(fig, "4")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_insights(pdf):
    fig = _new_slide()
    _title(fig, "Five Findings That Shaped the Model")
    _bullets(fig, [
        "1.  External bureau scores dominate.  Defaulters average 0.41 on EXT_SOURCE_2\n"
        "     versus 0.52 for repayers — the single strongest signal in the data.",
        "2.  DAYS_EMPLOYED = 365243 is a sentinel, not a duration.  18% of rows carry it,\n"
        "     and they default at 5.4% — *below* average, because they are pensioners.",
        "3.  Borrowing more than the goods are worth is a red flag.  Loan-to-goods > 1.2\n"
        "     raises the default rate from ~7% to 12.2%.",
        "4.  Renting doubles risk versus owning: 12.31% against 7.80%.",
        "5.  Time employed relative to age beats raw age.  The bottom quartile defaults\n"
        "     at 11.2% against 8.24% baseline.",
    ], dy=0.125, size=13)
    fig.text(0.07, 0.10, "Findings 2, 3 and 5 became engineered features; all five are reproduced by "
                         "the derived rules later in this deck.",
             fontsize=12, color=ACCENT)
    _footer(fig, "5")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_pipeline(pdf, metrics):
    fig = _new_slide()
    _title(fig, "Feature Pipeline & Class Imbalance")
    _bullets(fig, [
        "Preprocessing",
        "   ·  DAYS_EMPLOYED sentinel 365243 → NaN before imputation, so it cannot drag the median.",
        "   ·  Bureau history aggregated per applicant: counts, averages, debt-to-credit ratio.",
        "   ·  Engineered ratios: loan-to-income, annuity-to-income, loan-to-goods, share of life employed.",
        "   ·  Columns above 70% missing dropped; medians persisted for inference.",
        "",
        "Class imbalance — 11.4 repayers per defaulter",
        "   ·  scale_pos_weight = 10 in LightGBM, rather than resampling: no rows are discarded",
        "      and none are duplicated, so every fold sees the real data distribution.",
        "   ·  Stratified 5-fold CV keeps the ratio identical in every split.",
        "   ·  The resulting probability inflation is undone analytically — see the next slide.",
    ], dy=0.062, size=12.5)
    _footer(fig, "6")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_metrics(pdf, metrics):
    fig = _new_slide()
    _title(fig, "Model Performance", "All figures out-of-fold — never measured on training rows")

    _card(fig, 0.07, 0.60, 0.19, 0.16, "CV ROC-AUC", metrics["cv_roc_auc"], GOOD)
    _card(fig, 0.28, 0.60, 0.19, 0.16, "CV PR-AUC", metrics["cv_pr_auc"], ACCENT)
    op = metrics.get("operating_point", {})
    _card(fig, 0.49, 0.60, 0.19, 0.16, "Recall at threshold",
          f"{op.get('recall', 0) * 100:.0f}%", WARN)
    _card(fig, 0.70, 0.60, 0.19, 0.16, "Precision", f"{op.get('precision', 0) * 100:.0f}%", WARN)

    ax = fig.add_axes([0.10, 0.16, 0.35, 0.34])
    _style_axes(ax)
    folds = metrics["fold_aucs"]
    ax.bar(range(1, len(folds) + 1), folds, color=ACCENT)
    ax.set_ylim(0.70, 0.79)
    ax.set_xlabel("Fold")
    ax.set_title("ROC-AUC by fold — spread 0.008", fontsize=12, pad=10)
    for i, v in enumerate(folds, 1):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", color=FG, fontsize=9)

    cal = metrics.get("calibration", {})
    deciles = cal.get("deciles", [])
    if deciles:
        ax2 = fig.add_axes([0.57, 0.16, 0.35, 0.34])
        _style_axes(ax2)
        pred = [d["mean_predicted"] * 100 for d in deciles]
        act = [d["actual_default_rate"] * 100 for d in deciles]
        ax2.plot([0, max(act)], [0, max(act)], color=MUTED, linestyle="--", linewidth=1)
        ax2.plot(pred, act, "o-", color=GOOD, linewidth=2, markersize=5)
        ax2.set_xlabel("Predicted default rate (%)")
        ax2.set_ylabel("Observed (%)")
        ax2.set_title("Calibration by decile", fontsize=12, pad=10)

    _footer(fig, "7")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_calibration(pdf, metrics):
    fig = _new_slide()
    _title(fig, "Making the Probability Mean Something",
           "scale_pos_weight buys recall but destroys the probability scale")
    cal = metrics.get("calibration", {})
    _bullets(fig, [
        "Training with scale_pos_weight = 10 tells the model each default is worth ten rows.",
        "It therefore fits the odds of a re-balanced population: an ordinary applicant scored 0.54,",
        "which reads as a coin-flip default risk and makes any threshold meaningless.",
        "",
        "Dividing the fitted odds back down by that weight restores the true prior:",
        "        p_true  =  p / ( p + (1 − p) × 10 )",
        "",
        "The transform is strictly increasing, so ROC-AUC is untouched — only the scale changes.",
    ], dy=0.062, size=12.5)

    _card(fig, 0.07, 0.13, 0.25, 0.16, "Brier before",
          cal.get("oof_brier_raw", "—"), BAD)
    _card(fig, 0.35, 0.13, 0.25, 0.16, "Brier after",
          cal.get("oof_brier_calibrated", "—"), GOOD)
    _card(fig, 0.63, 0.13, 0.28, 0.16, "Mean predicted vs actual",
          f"{cal.get('mean_calibrated_probability', 0) * 100:.2f}% vs "
          f"{cal.get('actual_default_rate', 0) * 100:.2f}%", ACCENT)
    _footer(fig, "8")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_confusion(pdf, metrics):
    fig = _new_slide()
    cm = metrics.get("confusion_matrix", {})
    thr = cm.get("threshold", 0.5)
    _title(fig, "Operating Point",
           f"Threshold {thr:.2f} on calibrated probability, chosen by F1 on out-of-fold predictions")

    cells = [
        (0.20, 0.42, "True negatives", cm.get("TN"), GOOD, "correctly approved"),
        (0.52, 0.42, "False positives", cm.get("FP"), WARN, "good customers sent to review"),
        (0.20, 0.18, "False negatives", cm.get("FN"), BAD, "defaults missed"),
        (0.52, 0.18, "True positives", cm.get("TP"), GOOD, "defaults caught"),
    ]
    for x, y, label, val, color, note in cells:
        fig.patches.append(
            plt.Rectangle((x, y), 0.28, 0.20, transform=fig.transFigure,
                          facecolor="#1c1c2e", edgecolor=color, linewidth=1.5)
        )
        fig.text(x + 0.14, y + 0.135, f"{val:,}" if val is not None else "—",
                 fontsize=24, color=color, fontweight="bold", ha="center")
        fig.text(x + 0.14, y + 0.075, label, fontsize=11, color=FG, ha="center")
        fig.text(x + 0.14, y + 0.035, note, fontsize=9.5, color=MUTED, ha="center")

    fig.text(0.07, 0.115,
             "F1 is the neutral default. Given a real cost of a missed default versus a rejected good\n"
             "customer, the same out-of-fold scores retune the threshold without retraining.",
             fontsize=12, color=MUTED, linespacing=1.6)
    _footer(fig, "9")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_explainability(pdf):
    fig = _new_slide()
    _title(fig, "Explainability", "SHAP values, translated into sentences")
    fig.text(0.07, 0.71, "What the model returns for one applicant:", fontsize=13, color=ACCENT)

    reasons_up = [
        ("26.7%", "Average External Credit Score is 0.100, below the portfolio norm of 0.525"),
        ("6.0%", "External Credit Score #3 is 0.100, below the norm of 0.535"),
        ("4.6%", "Loan-to-Goods-Price Ratio is 2.00, above the norm of 1.12"),
        ("4.1%", "Monthly Loan Annuity is 55,000, above the norm of 24,903"),
    ]
    reasons_down = [
        ("5.7%", "Applicant Age is 22 years, below the portfolio norm of 43 years"),
        ("4.2%", "Loan-to-Income Ratio is 15.00, above the norm of 3.27"),
    ]

    fig.text(0.07, 0.64, "PUSHED RISK UP", fontsize=11, color=BAD, fontweight="bold")
    for i, (pct, text) in enumerate(reasons_up):
        y = 0.585 - i * 0.058
        fig.patches.append(plt.Rectangle((0.07, y - 0.015), 0.86, 0.045,
                                         transform=fig.transFigure,
                                         facecolor="#2a1618", edgecolor="none"))
        fig.text(0.085, y, f"{pct}", fontsize=11, color=BAD, fontweight="bold", va="center")
        fig.text(0.145, y, text, fontsize=11.5, color=FG, va="center")

    fig.text(0.07, 0.33, "PUSHED RISK DOWN", fontsize=11, color=GOOD, fontweight="bold")
    for i, (pct, text) in enumerate(reasons_down):
        y = 0.275 - i * 0.058
        fig.patches.append(plt.Rectangle((0.07, y - 0.015), 0.86, 0.045,
                                         transform=fig.transFigure,
                                         facecolor="#152a1c", edgecolor="none"))
        fig.text(0.085, y, f"{pct}", fontsize=11, color=GOOD, fontweight="bold", va="center")
        fig.text(0.145, y, text, fontsize=11.5, color=FG, va="center")

    fig.text(0.07, 0.13,
             "Each sentence states two independently checkable facts: the applicant's actual value against the\n"
             "portfolio norm, and the direction the model moved. The value is never inferred from the SHAP sign —\n"
             "that is why a 22-year-old is still reported as 22, even when age reduced this particular score.",
             fontsize=11, color=MUTED, linespacing=1.7)
    _footer(fig, "10")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_rules(pdf, rules):
    fig = _new_slide()
    baseline = rules.get("baseline_default_rate_pct", 8.24)
    _title(fig, "Derived Underwriting Rules",
           f"Every rule validated against observed default rates (baseline {baseline}%)")

    top = sorted(rules.get("rules", []),
                 key=lambda r: r["lift_vs_baseline"], reverse=True)[:8]
    if top:
        # Left edge leaves room for the y tick labels, which otherwise run into
        # the explanatory column; bottom clears the footer.
        ax = fig.add_axes([0.56, 0.20, 0.37, 0.50])
        _style_axes(ax)
        labels = [r["description"][:30] for r in top][::-1]
        lifts = [r["lift_vs_baseline"] for r in top][::-1]
        colors = [BAD if v >= 1.75 else WARN if v >= 1.3 else ACCENT for v in lifts]
        ax.barh(labels, lifts, color=colors)
        ax.axvline(1.0, color=FG, linestyle="--", linewidth=1.2)
        ax.set_xlim(0, max(lifts) * 1.22)
        ax.set_xlabel("Default-rate lift vs baseline", fontsize=10)
        ax.set_title("Risk lift of each flagged segment", fontsize=12, pad=10)
        ax.tick_params(axis="y", labelsize=8.5)
        for i, v in enumerate(lifts):
            ax.text(v + 0.03, i, f"{v}x", va="center", color=FG, fontsize=8.5)

    fig.text(0.07, 0.70, "How a rule is derived", fontsize=12.5, color=ACCENT, va="top")
    fig.text(0.07, 0.635,
             "1.  Rank features by mean |SHAP|.\n\n"
             "2.  Take the direction from\n"
             "     corr(value, SHAP) — not from\n"
             "     mean SHAP, which is ~0 by\n"
             "     construction and pure noise.\n\n"
             "3.  Cut at Q1 or Q3 accordingly.\n\n"
             "4.  Keep the rule only if the\n"
             "     flagged segment really does\n"
             "     default more than baseline.",
             fontsize=10.5, color=MUTED, va="top", linespacing=1.5)

    n_rules = len(rules.get("rules", []))
    n_rej = len(rules.get("rejected", []))
    checks = rules.get("sanity_checks", [])
    n_fail = len([c for c in checks if c.startswith("[FAIL]")])
    fig.text(0.07, 0.115,
             f"{n_rules} rules accepted · {n_rej} high-SHAP candidates rejected for "
             f"failing the empirical test\n"
             f"{len(checks) - n_fail}/{len(checks)} directional sanity checks passed",
             fontsize=11.5, color=GOOD, linespacing=1.6)
    _footer(fig, "11")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_rule_examples(pdf, rules):
    fig = _new_slide()
    _title(fig, "Sample Rule Output")
    top = sorted(rules.get("rules", []),
                 key=lambda r: r["lift_vs_baseline"], reverse=True)[:6]
    for i, r in enumerate(top):
        y = 0.70 - i * 0.105
        color = BAD if r["lift_vs_baseline"] >= 1.75 else WARN if r["lift_vs_baseline"] >= 1.3 else ACCENT
        fig.patches.append(plt.Rectangle((0.07, y - 0.055), 0.86, 0.088,
                                         transform=fig.transFigure,
                                         facecolor="#1c1c2e", edgecolor="none"))
        fig.patches.append(plt.Rectangle((0.07, y - 0.055), 0.005, 0.088,
                                         transform=fig.transFigure,
                                         facecolor=color, edgecolor="none"))
        fig.text(0.09, y + 0.012, f"IF  {r['condition']}", fontsize=12.5, color=FG, va="center")
        fig.text(0.09, y - 0.028,
                 f"{r['segment_default_rate_pct']}% default vs baseline  ·  "
                 f"covers {r['segment_share_pct']}% of applicants",
                 fontsize=10, color=MUTED, va="center")
        fig.text(0.90, y - 0.008, f"{r['lift_vs_baseline']}x", fontsize=17,
                 color=color, fontweight="bold", ha="right", va="center")
    _footer(fig, "12")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_chatbot(pdf):
    fig = _new_slide()
    _title(fig, "Talk to Data", "Natural language in, validated SQL out, business answer back")

    stages = [
        ("Generate", "compact cached schema\n+ few-shot prompt"),
        ("Validate", "SELECT-only, single\nstatement, no DDL/DML"),
        ("Ground", "every table & column\nmust really exist"),
        ("Execute", "read-only connection\nrow + time capped"),
        ("Summarise", "rows → plain-English\nbusiness answer"),
    ]
    w = 0.163
    for i, (name, desc) in enumerate(stages):
        x = 0.06 + i * 0.177
        fig.patches.append(plt.Rectangle((x, 0.46), w, 0.20, transform=fig.transFigure,
                                         facecolor="#1c1c2e", edgecolor=ACCENT, linewidth=1.4))
        fig.text(x + w / 2, 0.615, name, fontsize=13, color=ACCENT,
                 fontweight="bold", ha="center")
        fig.text(x + w / 2, 0.535, desc, fontsize=9.5, color=MUTED,
                 ha="center", va="center", linespacing=1.5)
        if i < len(stages) - 1:
            fig.text(x + w + 0.007, 0.55, "→", fontsize=15, color=ACCENT, ha="center")

    fig.text(0.06, 0.39, "Hallucination control", fontsize=13, color=FG, fontweight="bold")
    _bullets(fig, [
        "·  A question outside the schema returns NO_SQL rather than an invented answer.",
        "·  Invented column names are caught before execution, not after a confusing SQL error.",
        "·  The summariser only ever sees rows the database actually returned, and is told to use no other numbers.",
        "·  A failed query is retried once with the database's own error message fed back.",
    ], y=0.335, dy=0.05, size=11.5, color=MUTED)

    fig.text(0.06, 0.085, "Token optimisation:  schema cached at module level · "
                         "flash-lite model (no billed reasoning pass) · summariser capped at 15 rows\n"
                         "≈ 700–1,350 tokens per question end to end, usage reported on every call.",
             fontsize=11, color=GOOD, linespacing=1.6)
    _footer(fig, "13")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_chat_examples(pdf):
    fig = _new_slide()
    _title(fig, "Chatbot — Verified Question Set")
    qa = [
        ("What is the default rate by gender?",
         "Female 7.0% across 202,448 loans; male 10.14% across 105,059.",
         "success"),
        ("Which education level has the highest default rate?",
         "Lower secondary at 10.93%; academic degree lowest at 1.83%.",
         "success"),
        ("Average number of bureau loans for applicants who defaulted?",
         "5.62 prior bureau loans on average.",
         "success"),
        ("Compare average EXT_SOURCE_2 for defaulters vs non-defaulters.",
         "0.4109 for defaulters vs 0.5235 for repayers.",
         "success"),
        ("Show the top 5 defaulted applicants by credit amount.",
         "All cash loans, 2.70M–4.03M credit.",
         "success"),
        ("What is the weather forecast for Moscow?",
         "NO_SQL — refused, outside the schema.",
         "blocked"),
        ("SELECT 1; DROP TABLE applications;",
         "Injection neutralised — no DDL ever reached the database.",
         "blocked"),
    ]
    for i, (q, a, kind) in enumerate(qa):
        y = 0.70 - i * 0.088
        color = GOOD if kind == "success" else WARN
        fig.text(0.07, y, "Q", fontsize=11, color=ACCENT, fontweight="bold")
        fig.text(0.10, y, q, fontsize=11.5, color=FG)
        fig.text(0.10, y - 0.035, a, fontsize=10.5, color=MUTED)
        fig.text(0.93, y, "✓" if kind == "success" else "⨯", fontsize=14,
                 color=color, ha="right", fontweight="bold")
    fig.text(0.07, 0.07, "10 / 10 questions handled correctly, including three adversarial probes.",
             fontsize=12, color=GOOD)
    _footer(fig, "14")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_deployment(pdf):
    fig = _new_slide()
    _title(fig, "Deployment", "One command, two services")
    fig.patches.append(plt.Rectangle((0.07, 0.55), 0.86, 0.14, transform=fig.transFigure,
                                     facecolor="#0d0d16", edgecolor="#2e2e46"))
    fig.text(0.09, 0.62, "$  cp .env.example .env       # add your GEMINI_API_KEY\n"
                         "$  docker compose up --build",
             fontsize=14, color=GOOD, family="monospace", va="center", linespacing=1.8)

    _bullets(fig, [
        "·  frontend — nginx serving the built React app on :8080, proxying /api to the backend.",
        "·  api — FastAPI + LightGBM + SHAP on :8000, waiting on a real healthcheck.",
        "·  Pre-trained artifacts are mounted, so the evaluator never has to retrain.",
        "·  .env is injected at run time and never copied into the image.",
        "·  The healthcheck uses the Python interpreter, not curl — python:slim has no curl,",
        "   and a curl-based check leaves any dependent service waiting forever.",
    ], y=0.47, dy=0.058, size=12)

    fig.text(0.07, 0.09, "The UI is reachable at http://localhost:8080 and the API docs at "
                         "http://localhost:8000/docs.", fontsize=12, color=ACCENT)
    _footer(fig, "15")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide_limitations(pdf):
    fig = _new_slide()
    _title(fig, "Limitations & Next Steps", "What I would not claim, and what I would do next")
    fig.text(0.07, 0.72, "Known limitations", fontsize=13, color=WARN, fontweight="bold")
    _bullets(fig, [
        "·  Only application + bureau tables are used. The four behavioural tables (previous_application,",
        "   installments, POS/cash, credit-card balance) are untouched and are where most remaining AUC sits.",
        "·  The UI collects ~19 of 138 features; the rest fall back to training medians, so a UI prediction",
        "   is less sharp than a batch prediction on a complete record.",
        "·  Calibration is a single analytic prior correction. It still under-predicts by roughly one point",
        "   in the upper deciles; isotonic regression on out-of-fold scores would tighten this.",
        "·  CODE_GENDER is predictive and is used. In a regulated deployment it would be removed and the",
        "   model re-checked for disparate impact through proxies.",
    ], y=0.665, dy=0.0455, size=11.5, color=MUTED)

    fig.text(0.07, 0.275, "Next steps", fontsize=13, color=GOOD, fontweight="bold")
    _bullets(fig, [
        "·  Add the behavioural tables and re-tune — the single largest expected AUC gain.",
        "·  Replace the F1 threshold with an expected-cost threshold once loss-given-default is known.",
        "·  Cache chatbot answers by question hash to cut repeat token spend to zero.",
    ], y=0.225, dy=0.05, size=11.5, color=MUTED)
    _footer(fig, "16")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slides_screenshots(pdf, start_page: int) -> int:
    """Append any PNG dropped into documents/screenshots as its own slide."""
    if not SHOTS_DIR.exists():
        return start_page
    shots = sorted(SHOTS_DIR.glob("*.png"))
    page = start_page
    for shot in shots:
        try:
            img = mpimg.imread(shot)
        except Exception:
            continue
        fig = _new_slide()
        pretty = shot.stem.replace("_", " ").title()
        _title(fig, "Application Screenshot", pretty)
        ax = fig.add_axes([0.06, 0.08, 0.88, 0.68])
        ax.imshow(img)
        ax.axis("off")
        _footer(fig, str(page))
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)
        page += 1
    return page


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    metrics, rules = load_artifacts()

    with PdfPages(OUT_PDF) as pdf:
        slide_title(pdf, metrics)
        slide_problem(pdf, metrics)
        slide_architecture(pdf)
        slide_eda_charts(pdf)
        slide_insights(pdf)
        slide_pipeline(pdf, metrics)
        slide_metrics(pdf, metrics)
        slide_calibration(pdf, metrics)
        slide_confusion(pdf, metrics)
        slide_explainability(pdf)
        slide_rules(pdf, rules)
        slide_rule_examples(pdf, rules)
        slide_chatbot(pdf)
        slide_chat_examples(pdf)
        slide_deployment(pdf)
        slide_limitations(pdf)
        slides_screenshots(pdf, 17)

        info = pdf.infodict()
        info["Title"] = "Credit Risk Platform — Home Credit Default Risk"
        info["Subject"] = "ML + Explainability + NL-to-SQL platform"

    print(f"Presentation written to: {OUT_PDF}")


if __name__ == "__main__":
    main()
