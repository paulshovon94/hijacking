"""Emit the LaTeX body of the combined results table straight from the sweep CSV.

Generating rather than transcribing keeps the table and the raw results in sync and
removes the one error mode a hand-built table always has.

    python make_results_table.py ../wmt_code/results/seed_sweep_results.csv

Paste the output between the \\midrule and \\bottomrule of the results table in
translation_results.tex. The two are checked for drift, so regenerate rather than
editing a cell by hand.
"""
import csv
import sys
from collections import defaultdict
from statistics import mean, pstdev

rows = list(csv.DictReader(open(sys.argv[1])))

agg = defaultdict(list)
for r in rows:
    for metric in ("row_acc", "row_f1", "vote_acc", "vote_f1"):
        if r[metric] != "nan":
            agg[(r["split"], r["model"], r["head"], metric)].append(float(r[metric]))


def cell(split, model, head, metric):
    vals = agg[(split, model, head, metric)]
    if not vals:
        return None
    return 100 * mean(vals), 100 * pstdev(vals)


HEADS = [
    ("model_family", "Model Family", 25.00),
    ("learning_rate", "Learning Rate", 33.33),
    ("lora_r", "LoRA Rank, $r$", 33.33),
    ("lora_alpha", "LoRA Alpha, $\\alpha$", 33.33),
    ("lora_dropout", "LoRA Dropout", 50.00),
]

SPLITS = [
    ("row", "(a) Row-level split \\textit{(leaky -- shown for reference only)}"),
    ("shadow", "(b) Shadow-model split \\textit{(172 train / 44 unseen test models)}"),
    ("twin", "(c) Twin-aware split \\textit{(86 train / 22 unseen test twin pairs)}"),
]

METRICS = ["row_acc", "row_f1", "vote_acc", "vote_f1"]

out = []
for si, (split, split_label) in enumerate(SPLITS):
    if si:
        out.append("    \\midrule")
    out.append(f"    \\multicolumn{{10}}{{l}}{{\\textbf{{{split_label}}}}} \\\\")
    out.append("    \\addlinespace[2pt]")
    for head, label, chance in HEADS:
        vals = {}
        for model in ("forest", "neural"):
            for metric in METRICS:
                vals[(model, metric)] = cell(split, model, head, metric)
        # Bold the better of the two methods -- but only where the winner actually beats
        # its random-guess baseline. Bolding 23.2 over 13.0 on a head where both are far
        # below chance would present a failure as a win.
        bold = {}
        for metric in METRICS:
            f, n = vals[("forest", metric)], vals[("neural", metric)]
            win = max((f, n), key=lambda v: v[0]) if (f and n) else None
            # Clear chance by more than one standard deviation to count as a win. A head
            # that beats its baseline by less than its own seed-to-seed spread has not
            # demonstrated anything, and bolding it would say otherwise.
            if f is None or n is None or win[0] - chance <= win[1]:
                bold[metric] = (False, False)
            elif abs(f[0] - n[0]) < 5e-3:
                bold[metric] = (True, True)
            else:
                bold[metric] = (f[0] > n[0], n[0] > f[0])

        def fmt(model, metric):
            v = vals[(model, metric)]
            if v is None:
                return "--"
            m, s = v
            # The row-level block is a single deterministic seed-42 run; a "+/- 0.00"
            # there would imply a variance estimate that was never measured.
            body = f"{m:.2f}" if split == "row" else f"{m:.2f} $\\pm$ {s:.2f}"
            is_bold = bold[metric][0 if model == "forest" else 1]
            return f"\\textbf{{{body}}}" if is_bold else body

        cells = [fmt(m, met) for m in ("forest", "neural") for met in METRICS]
        out.append(f"    {label}")
        out.append(f"    & {chance:.2f}\\%")
        out.append("    & " + " & ".join(cells[:4]))
        out.append("    & " + " & ".join(cells[4:]) + " \\\\")
    out.append("")

print("\n".join(out))
