#!/usr/bin/env python3
"""
dts_pipeline.py -- single analysis pipeline for Paper 1 (draft v5).

Implements Section 4 (Methods) of the draft exactly:
  * cleaning: sentinel -99 rows removed, DT1/DT2 kept in 15-40 C,
    records split at Millis gaps/resets
  * estimator: 10 s windows, Delta T = DT1 - DT2 demeaned per window,
    C_T^2 = var / R^(2/3), C_n^2 = (79e-6 P / T^2)^2 C_T^2
  * run = independent unit: run medians, run-bootstrap CIs,
    exact permutation test on run medians, run-level Cliff's delta
  * window-level Mann-Whitney + Cliff's delta as secondary descriptors
  * TKE, mean offset, Spearman (within max), Welch PSDs
  * June 15 reported with and without Max1

Usage
-----
  python dts_pipeline.py                  # full analysis (edit CONFIG first)
  python dts_pipeline.py --tau step.csv   # probe time constant from a step test
  python dts_pipeline.py --check-phase    # reproduce the 17% factor and the
                                          # Rodriguez Garcia C_n^2 bounds (no data)

Outputs (in OUT_DIR)
--------------------
  paper_numbers.txt   every number the draft needs, grouped by section
  run_table.csv       one row per run
  numbers.csv         every reported statistic, long format
  figures/            PNGs named to match the draft's \\figfile paths
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal, stats

# =====================================================================
# CONFIG -- edit these before running
# =====================================================================
DATA_DIR = r"C:\Users\Otter\Downloads\Turbulence Python Code\drone_scintillation_data"            # folder with all session CSVs
OUT_DIR = "results"

R_SEP = 0.381             # probe separation [m]  <-- CONFIRM per session geometry
WINDOW_S = 10.0           # estimator window [s]
MIN_FILL = 0.8            # keep windows with >= 80% of expected samples
T_BOUNDS = (15.0, 40.0)   # valid probe range [C]
SENTINEL = -99
FS_NOMINAL = 38.46        # Hz

N_BOOT = 10000
SEED = 0

# Sessions: glob patterns relative to DATA_DIR. The script prints what each
# pattern matched -- CHECK THAT OUTPUT against Table 1 before trusting results.
SESSIONS = [
    dict(name="29 May", r_cm=15.62,
         files={"Background": ["5_29_26/*Background*.csv"],
                "Hover":      ["5_29_26/*Hover*.csv"],
                "Max":        ["5_29_26/*Max*.csv"]},
         expected={"Background": 3, "Hover": 3, "Max": 2}),
    dict(name="2 June", r_cm=5.00,
         files={"Background": ["6_2_26/6_2_26_Background*.csv"],
                "Hover":      ["6_2_26/*Hover*.csv"],
                "Max":        ["6_2_26/*Max*.csv"]},
         expected={"Background": 6, "Hover": 6, "Max": 6}),
    dict(name="15 June", r_cm=12.94,
         files={"Background": ["6_15_26/6_15_26_Background*.csv"],
                "Hover":      ["6_15_26/*Hover*.csv"],
                "Max":        ["6_15_26/*Max*.csv"]},
         expected={"Background": 3, "Hover": 6, "Max": 8},
         sensitivity_exclude=["_Max1.csv"]),
]

# Scintillation prediction (Section 5.6)
WAVELENGTH = 632.8e-9
WAKE_DZ = 0.30            # wake thickness along the beam [m]
L_MINUS_ZD = None         # distance from wake to receiver [m]  <-- FILL IN
PATH_L = 140.0
AMBIENT_SIGMA_R2 = None   # ambient path Rytov variance (camera or path C_n^2) <-- FILL IN
SAT_FACTOR = (3.4, 5.4)   # (R/L0)^(2/3) for L0 = 6..3 cm

# Tatarskii check (Section 5.4)
A2_TAT = 2.8
L0_WAKE = 0.05
MIXING_LAYER = 0.05

COLORS = {"Background": "#6c717a", "Hover": "#3aa55a", "Max": "#d9534f"}
CONDS = ["Background", "Hover", "Max"]


# =====================================================================
# Loading and cleaning
# =====================================================================
def load_run(path):
    """Read one DTS CSV, clean it, return list of contiguous segments."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    need = ["Millis", "DT1", "DT2"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{path}: missing column {c}; have {list(df.columns)}")
    wind = [c for c in ["WS", "u", "v", "w"] if c in df.columns]
    bad = np.zeros(len(df), bool)
    for c in wind:
        bad |= (df[c] <= SENTINEL + 0.5) & (df[c] >= SENTINEL - 0.5)
    for c in ["DT1", "DT2"]:
        bad |= (df[c] < T_BOUNDS[0]) | (df[c] > T_BOUNDS[1])
    df = df[~bad].reset_index(drop=True)

    # split at counter resets or gaps (> 5x nominal spacing)
    dms = df["Millis"].diff().to_numpy()
    step = 1000.0 / FS_NOMINAL
    breaks = np.where((dms <= 0) | (dms > 5 * step))[0]
    edges = [0] + list(breaks) + [len(df)]
    segs = [df.iloc[a:b].reset_index(drop=True) for a, b in zip(edges[:-1], edges[1:]) if b - a > 10]
    return segs


def cn2_prefactor(P_hPa, T_C):
    T = T_C + 273.15
    return (79e-6 * P_hPa / T**2) ** 2


def windows_from_segments(segs):
    """10 s windows: demeaned Delta-T variance -> C_n^2, plus mean offset."""
    out = []
    n_exp = WINDOW_S * FS_NOMINAL
    for s in segs:
        t = (s["Millis"] - s["Millis"].iloc[0]).to_numpy() / 1000.0
        idx = np.floor(t / WINDOW_S).astype(int)
        for k in np.unique(idx):
            w = s[idx == k]
            if len(w) < MIN_FILL * n_exp:
                continue
            dT = (w["DT1"] - w["DT2"]).to_numpy()
            var = np.var(dT - dT.mean())
            P = w["Pressure"].mean() if "Pressure" in w else 1005.0
            Tc = w["Temp"].mean() if "Temp" in w else 22.0
            ct2 = var / R_SEP ** (2 / 3)
            out.append(dict(cn2=cn2_prefactor(P, Tc) * ct2, offset=dT.mean(), P=P, Tc=Tc))
    return pd.DataFrame(out)


def run_summary(path, cond, session):
    segs = load_run(path)
    win = windows_from_segments(segs)
    full = pd.concat(segs, ignore_index=True)
    row = dict(session=session["name"], r_cm=session["r_cm"], cond=cond,
               file=os.path.basename(path), n_windows=len(win),
               duration_s=sum((s["Millis"].iloc[-1] - s["Millis"].iloc[0]) / 1000 for s in segs))
    if len(win):
        lc = np.log10(win["cn2"].to_numpy())
        row.update(cn2_median=np.median(win["cn2"]), offset_mean=win["offset"].mean(),
                   lag1_logcn2=pd.Series(lc).autocorr(1) if len(lc) > 2 else np.nan,
                   P_mean=win["P"].mean(), T_mean=win["Tc"].mean())
    if all(c in full for c in ["u", "v", "w"]):
        row["tke"] = 0.5 * (full["u"].var() + full["v"].var() + full["w"].var())
    if "WS" in full:
        row["ws_mean"] = full["WS"].mean()
    return row, win, segs


# =====================================================================
# Statistics
# =====================================================================
rng = np.random.default_rng(SEED)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def perm_test(a, b):
    """Exact two-sided permutation test on log10 run medians (mean difference)."""
    a, b = np.log10(a), np.log10(b)
    pooled = np.concatenate([a, b])
    na, n = len(a), len(pooled)
    obs = abs(a.mean() - b.mean())
    count = total = 0
    for comb in itertools.combinations(range(n), na):
        m = np.zeros(n, bool); m[list(comb)] = True
        stat = abs(pooled[m].mean() - pooled[~m].mean())
        count += stat >= obs - 1e-12
        total += 1
    # smallest attainable p: 2/total if group sizes are equal (the mirrored
    # assignment gives the same |stat|), otherwise 1/total
    p_min = (2 if len(a) == len(b) else 1) / total
    return count / total, p_min


def boot_median(x):
    x = np.asarray(x)
    bs = np.median(rng.choice(x, (N_BOOT, len(x)), replace=True), axis=1)
    return np.median(x), np.percentile(bs, [2.5, 97.5])


def boot_ratio(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ba = np.median(rng.choice(a, (N_BOOT, len(a))), axis=1)
    bb = np.median(rng.choice(b, (N_BOOT, len(b))), axis=1)
    return np.median(a) / np.median(b), np.percentile(ba / bb, [2.5, 97.5])


def compare(runs_a, runs_b, win_a, win_b):
    """All run-level and window-level statistics for condition a vs b."""
    ra, rb = runs_a["cn2_median"].to_numpy(), runs_b["cn2_median"].to_numpy()
    ratio, ci = boot_ratio(ra, rb)
    p_perm, p_min = perm_test(ra, rb)
    d_run = cliffs_delta(ra, rb)
    wa, wb = win_a["cn2"].to_numpy(), win_b["cn2"].to_numpy()
    p_mw = stats.mannwhitneyu(wa, wb, alternative="two-sided").pvalue
    d_win = cliffs_delta(wa, wb)
    detected = (ci[0] > 1) and (d_run > 0.33)
    return dict(ratio=ratio, ci_lo=ci[0], ci_hi=ci[1], p_perm=p_perm, p_min=p_min,
                delta_run=d_run, p_mw_window=p_mw, delta_window=d_win,
                n_a=len(ra), n_b=len(rb), detected=detected)


# =====================================================================
# Main analysis
# =====================================================================
def gather():
    runs, wins, psd_data = [], {}, {}
    for s in SESSIONS:
        print(f"\n=== {s['name']}  r = {s['r_cm']} cm ===")
        for cond in CONDS:
            files = sorted(set(f for pat in s["files"][cond]
                               for f in glob.glob(os.path.join(DATA_DIR, pat))))
            exp = s["expected"][cond]
            flag = "" if len(files) == exp else f"   <-- EXPECTED {exp}"
            print(f"  {cond:10s}: {len(files)} files{flag}")
            for f in files:
                print(f"      {os.path.basename(f)}")
                row, win, segs = run_summary(f, cond, s)
                row["sensitivity"] = any(os.path.basename(f).endswith(x)
                                         for x in s.get("sensitivity_exclude", []))
                runs.append(row)
                wins[(s["name"], cond, row["file"])] = win
                psd_data.setdefault((s["name"], cond), []).extend(segs)
    return pd.DataFrame(runs), wins, psd_data


def session_stats(rt, wins, sname, exclude_sensitivity=False):
    sub = rt[rt.session == sname]
    if exclude_sensitivity:
        sub = sub[~sub.sensitivity]
    def W(cond):
        keys = [(sname, cond, f) for f in sub[sub.cond == cond].file]
        return pd.concat([wins[k] for k in keys], ignore_index=True)
    res = {}
    for a, b in [("Max", "Background"), ("Hover", "Background"), ("Max", "Hover")]:
        A, B = sub[sub.cond == a].dropna(subset=["cn2_median"]), sub[sub.cond == b].dropna(subset=["cn2_median"])
        if len(A) >= 2 and len(B) >= 2:
            res[f"{a}_vs_{b}"] = compare(A, B, W(a), W(b))
    for cond in CONDS:
        c = sub[sub.cond == cond]
        if len(c):
            med, ci = boot_median(c["cn2_median"])
            res[f"{cond}_cn2"] = dict(median=med, ci_lo=ci[0], ci_hi=ci[1])
            if "tke" in c:
                res[f"{cond}_tke"] = dict(median=c["tke"].median())
            res[f"{cond}_offset"] = dict(median=c["offset_mean"].median())
    try:
        bw, mw = W("Background")["cn2"], W("Max")["cn2"]
        p95 = np.percentile(bw, 95)
        res["intermittency"] = dict(gamma=float((mw > p95).mean()), bkg_p95=p95, n_windows=len(mw))
    except ValueError:
        pass
    for cond in CONDS:
        c = sub[sub.cond == cond].dropna(subset=["ws_mean"]) if "ws_mean" in sub else []
        if len(c):
            res[f"{cond}_ws"] = dict(median=c["ws_mean"].median(), lo=c["ws_mean"].min(), hi=c["ws_mean"].max())
    mx = sub[sub.cond == "Max"].dropna(subset=["cn2_median", "ws_mean"])
    if len(mx) >= 4:
        rho, p = stats.spearmanr(mx["ws_mean"], mx["cn2_median"])
        res["spearman_max"] = dict(rho=rho, p=p, n=len(mx))
    return res


def implied_rms_dT(cn2, P=1005.0, Tc=22.0):
    return np.sqrt(cn2 / cn2_prefactor(P, Tc) * R_SEP ** (2 / 3))


def rytov_layer(cn2):
    k = 2 * np.pi / WAVELENGTH
    return 2.25 * k ** (7 / 6) * cn2 * WAKE_DZ * L_MINUS_ZD ** (5 / 6)


def psd_slope(f, P, fmin=0.5, fmax=5.0):
    m = (f >= fmin) & (f <= fmax) & (P > 0)
    if m.sum() < 3:
        return np.nan
    return np.polyfit(np.log10(f[m]), np.log10(P[m]), 1)[0]


def welch_segments(segs, col):
    fs, Ps = None, []
    for s in segs:
        x = (s["DT1"] - s["DT2"]).to_numpy() if col == "dT" else s[col].to_numpy()
        if len(x) < 512:
            continue
        f, P = signal.welch(x - x.mean(), fs=FS_NOMINAL, nperseg=512, noverlap=256)
        fs = f; Ps.append(P)
    return (fs, np.mean(Ps, axis=0)) if Ps else (None, None)


# =====================================================================
# Figures (names match the \figfile paths in draft v5)
# =====================================================================
def make_figures(rt, wins, allstats, psd_data, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "savefig.dpi": 300,
                         "savefig.bbox": "tight"})
    os.makedirs(figdir, exist_ok=True)
    qfloor = cn2_prefactor(1005, 22) * (2 * 0.01**2 / 12) / R_SEP ** (2 / 3)

    # Fig 2 detection: two radii nearest the wake
    near = sorted(SESSIONS, key=lambda s: s["r_cm"])[:2]
    fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, s in zip(axs, near):
        for i, cond in enumerate(CONDS):
            sub = rt[(rt.session == s["name"]) & (rt.cond == cond)]
            wv = np.concatenate([wins[(s["name"], cond, f)]["cn2"] for f in sub.file]) if len(sub) else []
            jit = rng.uniform(-0.18, 0.18, len(wv))
            ax.scatter(i + jit, np.log10(wv), s=4, alpha=0.35, color=COLORS[cond], lw=0)
            ax.scatter(np.full(len(sub), i), np.log10(sub.cn2_median), s=40,
                       color=COLORS[cond], edgecolor="k", zorder=3)
        ax.axhline(np.log10(qfloor), ls="--", color="0.5", lw=0.8)
        st = allstats[s["name"]]["all"]
        if "intermittency" in st:
            ax.axhline(np.log10(st["intermittency"]["bkg_p95"]), ls=":", color="k", lw=0.8)
            ax.text(2.45, np.log10(st["intermittency"]["bkg_p95"]), f" $\\gamma$={st['intermittency']['gamma']:.2f}",
                    fontsize=7, va="bottom", ha="right")
        for j, (key, lab) in enumerate([("Max_vs_Background", "max"), ("Hover_vs_Background", "hover")]):
            if key in st:
                c = st[key]
                ax.text(0.02, 0.97 - 0.07 * j,
                        f"{lab} vs bkg: p={c['p_perm']:.3f}, $\\delta$={c['delta_run']:+.2f}",
                        transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels(CONDS)
        ax.set_title(f"r = {s['r_cm']:.2f} cm ({s['name']})", fontsize=9)
    axs[0].set_ylabel(r"$\log_{10} C_n^2$ [m$^{-2/3}$]")
    fig.savefig(os.path.join(figdir, "slide_5_3_detection.png")); plt.close(fig)

    # Fig: run-median Cn2 by condition, all sessions
    fig, ax = plt.subplots(figsize=(6, 3))
    order = sorted(SESSIONS, key=lambda s: s["r_cm"])
    for gi, s in enumerate(order):
        st = allstats[s["name"]]["all"]
        for ci, cond in enumerate(CONDS):
            c = st.get(f"{cond}_cn2")
            if c:
                x = gi * 4 + ci
                ax.errorbar(x, c["median"], yerr=[[c["median"] - c["ci_lo"]], [c["ci_hi"] - c["median"]]],
                            fmt="o", color=COLORS[cond], mec="k", capsize=3)
        if "intermittency" in st:
            ax.hlines(st["intermittency"]["bkg_p95"], gi * 4 - 0.4, gi * 4 + 2.4, colors="k", linestyles=":", lw=0.8)
    ax.set_xticks([gi * 4 + 1 for gi in range(len(order))])
    ax.set_xticklabels([f"r = {s['r_cm']} cm\n({s['name']})" for s in order])
    for c in CONDS:
        ax.scatter([], [], color=COLORS[c], edgecolor="k", label=c)
    ax.set_yscale("log"); ax.set_ylabel(r"run-median $C_n^2$ [m$^{-2/3}$]"); ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(figdir, "slide_5_4_by_condition.png")); plt.close(fig)

    # Fig: Cn2 vs WS, max runs only
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for s, col in zip(order, ["C0", "C1", "C2"]):
        mx = rt[(rt.session == s["name"]) & (rt.cond == "Max")].dropna(subset=["ws_mean", "cn2_median"])
        if len(mx):
            sp = allstats[s["name"]]["all"].get("spearman_max")
            lab = f"r = {s['r_cm']} cm" + (f"  ($\\rho$={sp['rho']:.2f}, n={sp['n']})" if sp else "")
            ax.scatter(mx.ws_mean, mx.cn2_median, color=col, edgecolor="k", label=lab)
    ax.set_yscale("log"); ax.set_xlabel("run-mean WS [m s$^{-1}$]"); ax.set_ylabel(r"run-median $C_n^2$ [m$^{-2/3}$]")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(figdir, "slide_5_5_cn2_vs_ws.png")); plt.close(fig)

    # Fig 3 radial footprint
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for s in SESSIONS:
        c = allstats[s["name"]]["all"].get("Max_vs_Background")
        if c:
            ax.errorbar(s["r_cm"], c["ratio"], yerr=[[c["ratio"] - c["ci_lo"]], [c["ci_hi"] - c["ratio"]]],
                        fmt="o", color=COLORS["Max"], mec="k", capsize=3)
    ax.axvspan(6, 11, color="C0", alpha=0.12, label="predicted mixing layer")
    ax.axvline(8.4, ls="--", color="0.4", lw=0.8, label="slipstream edge")
    ax.axvline(11.9, ls=":", color="0.4", lw=0.8, label="blade tip")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_yscale("log"); ax.set_xlabel("r [cm]"); ax.set_ylabel("Max / background")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(figdir, "slide_5_6_radial.png")); plt.close(fig)

    # Fig 4 TKE vs Cn2
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    marks = {s["name"]: m for s, m in zip(SESSIONS, ["o", "s", "^"])}
    for _, r in rt.dropna(subset=["tke", "cn2_median"]).iterrows():
        ax.scatter(r.tke, r.cn2_median, color=COLORS[r.cond], marker=marks[r.session],
                   edgecolor="k", s=30)
    for s in SESSIONS:
        ax.scatter([], [], marker=marks[s["name"]], color="w", edgecolor="k", label=f"r = {s['r_cm']} cm")
    for c in CONDS:
        ax.scatter([], [], color=COLORS[c], label=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"TKE [m$^2$ s$^{-2}$]"); ax.set_ylabel(r"run-median $C_n^2$ [m$^{-2/3}$]")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(os.path.join(figdir, "tke_vs_cn2.png")); plt.close(fig)

    # Fig 5 mean offset per run
    fig, axs = plt.subplots(1, len(SESSIONS), figsize=(7, 2.8), sharey=False)
    for ax, s in zip(axs, sorted(SESSIONS, key=lambda s: s["r_cm"])):
        for i, cond in enumerate(CONDS):
            sub = rt[(rt.session == s["name"]) & (rt.cond == cond)]
            for j, f in enumerate(sub.file):
                w = wins[(s["name"], cond, f)]
                ax.errorbar(i + (j - len(sub) / 2) * 0.06, w["offset"].mean(), yerr=w["offset"].std(),
                            fmt="o", ms=4, color=COLORS[cond], capsize=2)
        bg = rt[(rt.session == s["name"]) & (rt.cond == "Background")].offset_mean.median()
        ax.axhline(bg, color="0.5", lw=0.7, ls="--")
        ax.set_xticks(range(3)); ax.set_xticklabels(["Bkg", "Hover", "Max"])
        ax.set_title(f"r = {s['r_cm']} cm", fontsize=9)
    axs[0].set_ylabel(r"mean $\Delta T$ [K]")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "offset.png")); plt.close(fig)

    # Fig 6 PSDs (session closest to the axis) -- saved under the draft's filename
    s = sorted(SESSIONS, key=lambda s: s["r_cm"])[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3))
    for cond in CONDS:
        f, P = welch_segments(psd_data.get((s["name"], cond), []), "dT")
        if f is not None:
            a1.loglog(f[1:], P[1:], color=COLORS[cond], label=cond)
        f, P = welch_segments(psd_data.get((s["name"], cond), []), "WS")
        if f is not None:
            a2.loglog(f[1:], P[1:], color=COLORS[cond], label=cond)
    ff = np.array([0.5, 5]); a1.loglog(ff, 1e-3 * ff ** (-5 / 3), "k--", lw=0.8, label="-5/3")
    a1.axhline(2 * 0.01**2 / 12 / (FS_NOMINAL / 2), color="0.5", ls=":", lw=0.8, label="quantization")
    a1.axvline(1 / WINDOW_S, color="0.6", lw=0.6)
    a1.set_xlabel("f [Hz]"); a1.set_ylabel(r"PSD of $\Delta T$ [K$^2$/Hz]"); a1.set_title("(a)", loc="left")
    a2.set_xlabel("f [Hz]"); a2.set_ylabel("PSD of WS [m$^2$ s$^{-2}$/Hz]"); a2.set_title("(b)", loc="left")
    a1.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "jun2_detection_and_psd.png")); plt.close(fig)


# =====================================================================
# Report
# =====================================================================
def fmt(x, sig=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{sig}g}"


def write_report(rt, allstats, psd_data, out):
    L, rows = [], []
    def add(section, name, value):
        rows.append(dict(section=section, quantity=name, value=value))

    L.append("PAPER NUMBERS  (draft v5; replace \\prov{} values)\n" + "=" * 60)
    L.append(f"R_SEP = {R_SEP} m  |  window = {WINDOW_S} s  |  bootstrap = {N_BOOT}\n")

    L.append("Table 1 -- run inventory")
    for s in SESSIONS:
        sub = rt[rt.session == s["name"]]
        L.append(f"  {s['name']:8s} r={s['r_cm']:5.2f}  " +
                 "  ".join(f"{c}={int((sub.cond == c).sum())}" for c in CONDS) +
                 f"  durations {sub.duration_s.min():.0f}-{sub.duration_s.max():.0f} s")
    L.append(f"  median lag-1 autocorr of window log10 Cn2: {rt.lag1_logcn2.median():.2f}\n")

    L.append("Sec 5.1 Detection / Table 2 -- max throttle vs same-session background")
    for s in sorted(SESSIONS, key=lambda s: s["r_cm"]):
        for tag, st in allstats[s["name"]].items():
            c = st.get("Max_vs_Background"); h = st.get("Hover_vs_Background"); mh = st.get("Max_vs_Hover")
            cm = st.get("Max_cn2")
            label = f"{s['r_cm']:5.2f} cm [{tag}]"
            if cm:
                L.append(f"  {label}  max Cn2 = {cm['median']:.2e} [{cm['ci_lo']:.2e}, {cm['ci_hi']:.2e}]")
            if c:
                L.append(f"      ratio = {c['ratio']:.2g} [{c['ci_lo']:.2g}, {c['ci_hi']:.2g}]  "
                         f"perm p = {c['p_perm']:.3f} (min {c['p_min']:.3f})  delta_run = {c['delta_run']:+.2f}  "
                         f"| window: MW p = {c['p_mw_window']:.1e}, delta = {c['delta_window']:+.2f}  "
                         f"DETECTED = {c['detected']}")
                for k, v in c.items():
                    add("5.1", f"{s['r_cm']}cm_{tag}_max_vs_bkg_{k}", v)
            if h:
                L.append(f"      hover vs bkg: ratio {h['ratio']:.2g} [{h['ci_lo']:.2g}, {h['ci_hi']:.2g}]  "
                         f"delta_run {h['delta_run']:+.2f}  window delta {h['delta_window']:+.2f}")
            if mh:
                L.append(f"      max vs hover: perm p = {mh['p_perm']:.3f}  delta_run {mh['delta_run']:+.2f}")
    L.append("")

    L.append("Sec 5.1 -- intermittency factor gamma (max windows above background p95)")
    for s in sorted(SESSIONS, key=lambda s: s["r_cm"]):
        g = allstats[s["name"]]["all"].get("intermittency")
        if g:
            L.append(f"  {s['r_cm']:5.2f} cm  gamma = {g['gamma']:.2f}  (bkg p95 {g['bkg_p95']:.2e}, {g['n_windows']} windows)")
    bk = {s["name"]: allstats[s["name"]]["all"]["Background_cn2"]["median"] for s in SESSIONS}
    L.append(f"Sec 5.2 -- background spread between sessions: max/min = {max(bk.values())/min(bk.values()):.2g}  {bk}\n")

    L.append("Sec 5.3 / Table (wind speed) -- run-mean WS [m/s]: median [min, max]")
    for s in sorted(SESSIONS, key=lambda s: s["r_cm"]):
        st = allstats[s["name"]]["all"]
        L.append(f"  {s['r_cm']:5.2f} cm  " + "  ".join(
            f"{c}: {st[c+'_ws']['median']:.2f} [{st[c+'_ws']['lo']:.2f}, {st[c+'_ws']['hi']:.2f}]"
            for c in CONDS if c + "_ws" in st))
    L.append("")

    L.append("Sec 5.4 / Table 3 -- median TKE [m2/s2] and hover/background TKE ratio")
    for s in sorted(SESSIONS, key=lambda s: s["r_cm"]):
        st = allstats[s["name"]]["all"]
        t = {c: st.get(f"{c}_tke", {}).get("median", np.nan) for c in CONDS}
        L.append(f"  {s['r_cm']:5.2f} cm  bkg {fmt(t['Background'])}  hover {fmt(t['Hover'])}  "
                 f"max {fmt(t['Max'])}   hover/bkg = {fmt(t['Hover'] / t['Background'])}")
        sp = st.get("spearman_max")
        if sp:
            L.append(f"      Spearman (max runs, Cn2 vs WS): rho = {sp['rho']:.2f}, p = {sp['p']:.3f}, n = {sp['n']}")
    L.append("")

    L.append("Sec 5.5 -- mean offset shift (max - background) and Tatarskii check")
    for s in sorted(SESSIONS, key=lambda s: s["r_cm"]):
        st = allstats[s["name"]]["all"]
        sh = st["Max_offset"]["median"] - st["Background_offset"]["median"]
        g1, g2 = abs(sh) / R_SEP, abs(sh) / MIXING_LAYER
        K = cn2_prefactor(1005, 22)
        L.append(f"  {s['r_cm']:5.2f} cm  shift = {sh:+.3f} K   grad(R) = {g1:.2f} K/m -> Cn2 {K*A2_TAT*L0_WAKE**(4/3)*g1**2:.1e}"
                 f"   grad(mixing layer) = {g2:.1f} K/m -> Cn2 {K*A2_TAT*L0_WAKE**(4/3)*g2**2:.1e}")
        add("5.5", f"{s['r_cm']}cm_offset_shift_K", sh)
    L.append("")

    L.append("Discussion -- implied rms Delta T at max throttle")
    for s in sorted(SESSIONS, key=lambda s: s["r_cm"]):
        cm = allstats[s["name"]]["all"]["Max_cn2"]["median"]
        L.append(f"  {s['r_cm']:5.2f} cm  {implied_rms_dT(cm):.3f} K")
    L.append("")

    L.append("Sec 5.6 -- Delta-T PSD slope 0.5-5 Hz")
    for s in sorted(SESSIONS, key=lambda s: s["r_cm"]):
        for cond in CONDS:
            f, P = welch_segments(psd_data.get((s["name"], cond), []), "dT")
            if f is not None:
                L.append(f"  {s['r_cm']:5.2f} cm {cond:10s} slope = {psd_slope(f, P):.2f}")
    L.append("")

    L.append("Sec 5.7 / Table (scintillation) -- predicted scintillation")
    if L_MINUS_ZD is None:
        L.append("  L_MINUS_ZD not set in CONFIG -- skipped")
    else:
        for s in sorted(SESSIONS, key=lambda s: s["r_cm"]):
            st = allstats[s["name"]]["all"]
            if st.get("Max_vs_Background", {}).get("detected"):
                d = rytov_layer(st["Max_cn2"]["median"])
                line = f"  {s['r_cm']:5.2f} cm  d_sigmaR2 = {d:.1e}   saturation-corrected {d*SAT_FACTOR[0]:.1e}-{d*SAT_FACTOR[1]:.1e}"
                if AMBIENT_SIGMA_R2:
                    line += f"   = {100*d/AMBIENT_SIGMA_R2:.2f}% ({100*d*SAT_FACTOR[1]/AMBIENT_SIGMA_R2:.1f}% corrected) of ambient"
                L.append(line)
    bkg = rt[rt.cond == "Background"].cn2_median.median()
    k = 2 * np.pi / WAVELENGTH
    L.append(f"  (reference: background probe Cn2 {bkg:.1e} applied uniformly over {PATH_L} m "
             f"-> sigma_R^2 = {1.23*bkg*k**(7/6)*PATH_L**(11/6):.3g})")

    with open(os.path.join(out, "paper_numbers.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(out, "numbers.csv"), index=False)
    print("\n".join(L))


def main_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)
    rt, wins, psd_data = gather()
    if rt.empty:
        sys.exit("No files matched -- check DATA_DIR and the patterns in SESSIONS.")
    rt.to_csv(os.path.join(OUT_DIR, "run_table.csv"), index=False)
    allstats = {}
    for s in SESSIONS:
        allstats[s["name"]] = {"all": session_stats(rt, wins, s["name"])}
        if s.get("sensitivity_exclude") and rt[(rt.session == s["name"])].sensitivity.any():
            allstats[s["name"]]["without_excluded_run"] = session_stats(rt, wins, s["name"], True)
    write_report(rt, allstats, psd_data, OUT_DIR)
    make_figures(rt, wins, allstats, psd_data, os.path.join(OUT_DIR, "figures"))
    print(f"\nWrote {OUT_DIR}/paper_numbers.txt, run_table.csv, numbers.csv, figures/")


# =====================================================================
# Probe time constant from a bench step test
# =====================================================================
def fit_tau(path, channel="DT1", fit_s=20.0):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    t = (df["Millis"] - df["Millis"].iloc[0]).to_numpy() / 1000.0
    x = df[channel].to_numpy()
    xs = pd.Series(x).rolling(5, center=True, min_periods=1).mean().to_numpy()
    i0 = int(np.argmax(np.abs(np.gradient(xs, t))))
    m = (t >= t[i0] - 3.0) & (t <= t[i0] + fit_s)
    tt, xx = t[m], x[m]
    from scipy.optimize import curve_fit
    def model(tt, Tf, T0, tau, t0):   # flat at T0 until onset t0, then exponential
        return np.where(tt < t0, T0, Tf + (T0 - Tf) * np.exp(-(tt - t0) / tau))
    p, cov = curve_fit(model, tt, xx, p0=[xx[-1], xx[0], 1.0, t[i0] - 0.2], maxfev=20000)
    tau, dtau = p[2], np.sqrt(cov[2, 2])
    print(f"{channel}: tau = {tau:.3f} +/- {dtau:.3f} s  -> -3 dB bandwidth = {1/(2*np.pi*tau):.2f} Hz")
    for U in (2.0, 5.0, 8.0):
        print(f"   resolved eddy scale U*tau at U = {U} m/s: {U*tau:.2f} m")
    print("Run several steps (warm and cool, both probes) and report the mean and spread.")


# =====================================================================
# Theory check: 17% factor and the Rodriguez Garcia bounds (no data)
# =====================================================================
def check_phase():
    from scipy.special import j1, jv
    from scipy.integrate import quad
    k = 2 * np.pi / WAVELENGTH
    D = 0.10
    F = lambda x: 1 - (2 * j1(x) / x) ** 2 - 16 * (jv(2, x) / x) ** 2
    def var(Cn2, L, L0=None):
        k0 = 0 if L0 is None else 2 * np.pi / L0
        g = lambda u: (np.exp(u) * 0.033 * (np.exp(2 * u) + k0**2) ** (-11 / 6)
                       * F(np.exp(u) * D / 2) * np.exp(u))
        return 4 * np.pi**2 * k**2 * L * Cn2 * quad(g, np.log(1e-2), np.log(1e5), limit=400)[0]
    r0 = (0.423 * k**2 * 1e-14 * 1.0) ** (-3 / 5)
    print(f"Kolmogorov check: numeric {var(1e-14, 1.0):.3e} vs 0.134(D/r0)^(5/3) = {0.134*(D/r0)**(5/3):.3e}")
    print(f"vK/Kolmogorov (piston+tilt removed), L0 = 5 cm, D = 10 cm: {var(1e-14,1,0.05)/var(1e-14,1):.3f}")
    L_jet = 0.54
    for sig in (0.006, 0.007):
        s2 = (2 * np.pi * sig) ** 2
        print(f"RMS {sig} wave: Cn2 bound Kolmogorov {s2/var(1,L_jet):.1e}  | L0=5cm {s2/var(1,L_jet,0.05):.1e}"
              f"  | L0=5cm surface units {4*s2/var(1,L_jet,0.05):.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", metavar="CSV", help="fit probe time constant from a step-test CSV")
    ap.add_argument("--channel", default="DT1")
    ap.add_argument("--check-phase", action="store_true")
    a = ap.parse_args()
    if a.tau:
        fit_tau(a.tau, a.channel)
    elif a.check_phase:
        check_phase()
    else:
        main_analysis()