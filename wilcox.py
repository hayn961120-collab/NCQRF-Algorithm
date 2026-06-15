# compare_mcqf_vs_grf_per_tau_wilcoxon.py

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("paired data 길이가 다릅니다.")

    diff = x - y
    nz = diff[np.abs(diff) > 0]

    if len(nz) == 0:
        return {
            "n_pairs": int(len(diff)),
            "n_nonzero_diff": 0,
            "wilcoxon_statistic": np.nan,
            "wilcoxon_pvalue": np.nan,
            "mean_diff": float(np.mean(diff)),
            "median_diff": float(np.median(diff)),
        }

    res = wilcoxon(x, y, zero_method="wilcox", correction=False, mode="auto")
    return {
        "n_pairs": int(len(diff)),
        "n_nonzero_diff": int(len(nz)),
        "wilcoxon_statistic": float(res.statistic),
        "wilcoxon_pvalue": float(res.pvalue),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }


def load_mcqf_from_compare_json(json_path: str):
    res = load_json(json_path)

    tau = np.array(res["tau"], dtype=float)
    tau = np.round(tau, 6)

    mcqf_comp = np.array(res["mcqf"]["comp"], dtype=float)
    mcqf_per_tau = np.array(res["mcqf"]["per_tau"], dtype=float)

    if mcqf_per_tau.ndim != 2:
        raise ValueError("mcqf['per_tau']는 2차원이어야 합니다.")

    n_mc, k = mcqf_per_tau.shape
    if len(tau) != k:
        raise ValueError(f"tau 길이({len(tau)})와 mcqf_per_tau 열 수({k})가 다릅니다.")

    comp_df = pd.DataFrame({
        "seed": np.arange(n_mc, dtype=int),
        "mcqf_comp": mcqf_comp
    })

    rows = []
    for i in range(n_mc):
        for j, t in enumerate(tau):
            rows.append({
                "seed": int(i),
                "tau": round(float(t), 6),
                "mcqf_pinball": float(mcqf_per_tau[i, j])
            })
    per_tau_df = pd.DataFrame(rows)

    return tau, comp_df, per_tau_df


def load_grf_raw(comp_csv_path: str, per_tau_csv_path: str, tau_expected=None):
    comp_df = pd.read_csv(comp_csv_path)
    per_tau_df = pd.read_csv(per_tau_csv_path)

    comp_df["seed"] = comp_df["seed"].astype(int)
    per_tau_df["seed"] = per_tau_df["seed"].astype(int)
    per_tau_df["tau"] = per_tau_df["tau"].astype(float).round(6)

    comp_df = comp_df.rename(columns={"test_comp": "grf_comp"})
    per_tau_df = per_tau_df.rename(columns={"test_pinball": "grf_pinball"})

    tau_grf = np.sort(per_tau_df["tau"].unique())

    if tau_expected is not None:
        tau_expected = np.round(np.array(tau_expected, dtype=float), 6)
        if not np.allclose(tau_grf, tau_expected):
            raise ValueError(f"GRF tau={tau_grf} 와 MCQF tau={tau_expected} 가 다릅니다.")

    return comp_df[["seed", "grf_comp"]], per_tau_df[["seed", "tau", "grf_pinball"]]

def summarize_per_tau_wilcoxon(mcqf_tau_df: pd.DataFrame, grf_tau_df: pd.DataFrame):
    merged = pd.merge(
        mcqf_tau_df,
        grf_tau_df,
        on=["seed", "tau"],
        how="inner"
    ).sort_values(["tau", "seed"]).reset_index(drop=True)

    if merged.empty:
        raise ValueError("MCQF와 GRF의 공통 seed-tau 조합이 없습니다.")

    out_rows = []

    for t in sorted(merged["tau"].unique()):
        sub = merged[merged["tau"] == t].copy()

        # diff = grf - mcqf (양수면 MCQF가 더 좋음)
        diff = sub["grf_pinball"].values - sub["mcqf_pinball"].values
        w = safe_wilcoxon(sub["grf_pinball"].values, sub["mcqf_pinball"].values)

        out_rows.append({
            "tau": float(t),
            "n_pairs": w["n_pairs"],
            "n_nonzero_diff": w["n_nonzero_diff"],
            "mcqf_mean_pinball": float(sub["mcqf_pinball"].mean()),
            "grf_mean_pinball": float(sub["grf_pinball"].mean()),
            "mean_diff_grf_minus_mcqf": float(diff.mean()),
            "median_diff_grf_minus_mcqf": float(np.median(diff)),
            "wilcoxon_statistic": w["wilcoxon_statistic"],
            "wilcoxon_pvalue": w["wilcoxon_pvalue"],
            "interpretation": "mean_diff_grf_minus_mcqf > 0 이면 MCQF가 더 좋음"
        })

    summary_df = pd.DataFrame(out_rows)
    return merged, summary_df


def summarize_composite_wilcoxon(mcqf_comp_df: pd.DataFrame, grf_comp_df: pd.DataFrame):
    merged = pd.merge(
        mcqf_comp_df,
        grf_comp_df,
        on="seed",
        how="inner"
    ).sort_values("seed").reset_index(drop=True)

    if merged.empty:
        raise ValueError("MCQF와 GRF의 공통 seed가 없습니다.")

    diff = merged["grf_comp"].values - merged["mcqf_comp"].values
    w = safe_wilcoxon(merged["grf_comp"].values, merged["mcqf_comp"].values)

    summary_df = pd.DataFrame([{
        "n_pairs": w["n_pairs"],
        "n_nonzero_diff": w["n_nonzero_diff"],
        "mcqf_mean_comp": float(merged["mcqf_comp"].mean()),
        "grf_mean_comp": float(merged["grf_comp"].mean()),
        "mean_diff_grf_minus_mcqf": float(diff.mean()),
        "median_diff_grf_minus_mcqf": float(np.median(diff)),
        "wilcoxon_statistic": w["wilcoxon_statistic"],
        "wilcoxon_pvalue": w["wilcoxon_pvalue"],
        "interpretation": "mean_diff_grf_minus_mcqf > 0 이면 MCQF가 더 좋음"
    }])

    return merged, summary_df


def save_excel(out_xlsx, per_tau_summary, comp_summary, per_tau_raw, comp_raw):
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)

    notes = pd.DataFrame({
        "note": [
            "per_tau_wilcoxon 시트: tau별 Wilcoxon signed-rank test 결과",
            "composite_wilcoxon 시트: composite pinball loss 기준 Wilcoxon 결과",
            "diff_grf_minus_mcqf = grf - mcqf",
            "diff_grf_minus_mcqf > 0 이면 MCQF loss가 더 작아서 MCQF가 더 좋음",
            "loss는 작을수록 좋음"
        ]
    })

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        per_tau_summary.to_excel(writer, index=False, sheet_name="per_tau_wilcoxon")
        comp_summary.to_excel(writer, index=False, sheet_name="composite_wilcoxon")
        per_tau_raw.to_excel(writer, index=False, sheet_name="per_tau_raw_merged")
        comp_raw.to_excel(writer, index=False, sheet_name="comp_raw_merged")
        notes.to_excel(writer, index=False, sheet_name="notes")


if __name__ == "__main__":
    mcqf_json_path = r"C:\quantile\0514_leaf1,2,5,10,20,25,30_n1000_result\grid_results_p1_s2\res_s2_p1.json"

    grf_comp_csv_path = r"C:\quantile\0423grf\n1000_param_leaf_1,2,5,10,20,25,30\fal_result_grf\raw\grf_s2_p1_comp_raw.csv"
    grf_tau_csv_path  = r"C:\quantile\0423grf\n1000_param_leaf_1,2,5,10,20,25,30\fal_result_grf\raw\grf_s2_p1_per_tau_raw.csv"

    out_xlsx = r"C:\quantile\all_result\excel\0514_wilcoxon_pinball_vs_grf_s2_p1.xlsx"

    tau, mcqf_comp_df, mcqf_tau_df = load_mcqf_from_compare_json(mcqf_json_path)
    grf_comp_df, grf_tau_df = load_grf_raw(
        grf_comp_csv_path,
        grf_tau_csv_path,
        tau_expected=tau
    )

    per_tau_raw, per_tau_summary = summarize_per_tau_wilcoxon(mcqf_tau_df, grf_tau_df)
    comp_raw, comp_summary = summarize_composite_wilcoxon(mcqf_comp_df, grf_comp_df)

    save_excel(
        out_xlsx=out_xlsx,
        per_tau_summary=per_tau_summary,
        comp_summary=comp_summary,
        per_tau_raw=per_tau_raw,
        comp_raw=comp_raw
    )

    print("[Saved Excel]", out_xlsx)
    print()
    print("[Per-tau Wilcoxon]")
    print(per_tau_summary.to_string(index=False))
    print()
    print("[Composite Wilcoxon]")
    print(comp_summary.to_string(index=False))
