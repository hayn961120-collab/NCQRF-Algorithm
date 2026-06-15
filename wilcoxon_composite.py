import os
import json
import itertools
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


def load_mcqf_from_json(json_path: str):
    res = load_json(json_path)

    tau = np.round(np.array(res["tau"], dtype=float), 6)

    comp = np.array(res["mcqf"]["comp"], dtype=float)
    per_tau = np.array(res["mcqf"]["per_tau"], dtype=float)

    if per_tau.ndim != 2:
        raise ValueError("mcqf['per_tau']는 2차원이어야 합니다.")

    n_mc, k = per_tau.shape
    if len(tau) != k:
        raise ValueError(f"tau 길이({len(tau)})와 mcqf_per_tau 열 수({k})가 다릅니다.")

    comp_df = pd.DataFrame({
        "seed": np.arange(n_mc, dtype=int),
        "mcqf_comp": comp
    })

    rows = []
    for i in range(n_mc):
        for j, t in enumerate(tau):
            rows.append({
                "seed": int(i),
                "tau": round(float(t), 6),
                "mcqf_pinball": float(per_tau[i, j])
            })
    per_tau_df = pd.DataFrame(rows)

    return tau, comp_df, per_tau_df


def load_qrf_from_json(json_path: str, tau_expected=None):
    res = load_json(json_path)

    tau = np.round(np.array(res["tau"], dtype=float), 6)

    # 네 기존 코드 기준으로 rfqr 키 사용
    comp = np.array(res["rfqr"]["comp"], dtype=float)
    per_tau = np.array(res["rfqr"]["per_tau"], dtype=float)

    if per_tau.ndim != 2:
        raise ValueError("rfqr['per_tau']는 2차원이어야 합니다.")

    n_rf, k = per_tau.shape
    if len(tau) != k:
        raise ValueError(f"tau 길이({len(tau)})와 rfqr_per_tau 열 수({k})가 다릅니다.")

    if tau_expected is not None:
        tau_expected = np.round(np.array(tau_expected, dtype=float), 6)
        if not np.allclose(tau, tau_expected):
            raise ValueError(f"QRF tau={tau} 와 기준 tau={tau_expected} 가 다릅니다.")

    comp_df = pd.DataFrame({
        "seed": np.arange(n_rf, dtype=int),
        "qrf_comp": comp
    })

    rows = []
    for i in range(n_rf):
        for j, t in enumerate(tau):
            rows.append({
                "seed": int(i),
                "tau": round(float(t), 6),
                "qrf_pinball": float(per_tau[i, j])
            })
    per_tau_df = pd.DataFrame(rows)

    return comp_df, per_tau_df


def load_grf_raw(comp_csv_path: str, per_tau_csv_path: str, tau_expected=None):
    comp_df = pd.read_csv(comp_csv_path)
    per_tau_df = pd.read_csv(per_tau_csv_path)

    comp_df["seed"] = comp_df["seed"].astype(int)
    per_tau_df["seed"] = per_tau_df["seed"].astype(int)
    per_tau_df["tau"] = per_tau_df["tau"].astype(float).round(6)

    # 기존 GRF raw 파일 컬럼명 기준
    comp_df = comp_df.rename(columns={"test_comp": "grf_comp"})
    per_tau_df = per_tau_df.rename(columns={"test_pinball": "grf_pinball"})

    tau_grf = np.sort(per_tau_df["tau"].unique())

    if tau_expected is not None:
        tau_expected = np.round(np.array(tau_expected, dtype=float), 6)
        if not np.allclose(tau_grf, tau_expected):
            raise ValueError(f"GRF tau={tau_grf} 와 기준 tau={tau_expected} 가 다릅니다.")

    return comp_df[["seed", "grf_comp"]], per_tau_df[["seed", "tau", "grf_pinball"]]


def pairwise_wilcoxon_composite(all_comp_df: pd.DataFrame, model_cols: dict):
    """
    all_comp_df: seed + model comp columns merged dataframe
    model_cols: {"mcqf": "mcqf_comp", "qrf": "qrf_comp", "grf": "grf_comp"}
    """
    results = []

    for m1, m2 in itertools.combinations(model_cols.keys(), 2):
        c1 = model_cols[m1]
        c2 = model_cols[m2]

        sub = all_comp_df[["seed", c1, c2]].dropna().sort_values("seed").reset_index(drop=True)

        if sub.empty:
            continue

        # diff = model2 - model1 (양수면 model1이 더 좋음; loss 작을수록 좋음)
        diff = sub[c2].values - sub[c1].values
        w = safe_wilcoxon(sub[c2].values, sub[c1].values)

        results.append({
            "model_1": m1,
            "model_2": m2,
            "n_pairs": w["n_pairs"],
            "n_nonzero_diff": w["n_nonzero_diff"],
            f"{m1}_mean_comp": float(sub[c1].mean()),
            f"{m2}_mean_comp": float(sub[c2].mean()),
            f"mean_diff_{m2}_minus_{m1}": float(diff.mean()),
            f"median_diff_{m2}_minus_{m1}": float(np.median(diff)),
            "wilcoxon_statistic": w["wilcoxon_statistic"],
            "wilcoxon_pvalue": w["wilcoxon_pvalue"],
            "interpretation": f"mean_diff_{m2}_minus_{m1} > 0 이면 {m1}가 더 좋음"
        })

    return pd.DataFrame(results)


def pairwise_wilcoxon_per_tau(all_tau_df: pd.DataFrame, pinball_cols: dict):
    """
    all_tau_df: seed, tau + model pinball columns merged dataframe
    pinball_cols: {"mcqf": "mcqf_pinball", "qrf": "qrf_pinball", "grf": "grf_pinball"}
    """
    results = []

    for tau_value in sorted(all_tau_df["tau"].dropna().unique()):
        tau_sub = all_tau_df[all_tau_df["tau"] == tau_value].copy()

        for m1, m2 in itertools.combinations(pinball_cols.keys(), 2):
            c1 = pinball_cols[m1]
            c2 = pinball_cols[m2]

            sub = tau_sub[["seed", "tau", c1, c2]].dropna().sort_values("seed").reset_index(drop=True)

            if sub.empty:
                continue

            diff = sub[c2].values - sub[c1].values
            w = safe_wilcoxon(sub[c2].values, sub[c1].values)

            results.append({
                "tau": float(tau_value),
                "model_1": m1,
                "model_2": m2,
                "n_pairs": w["n_pairs"],
                "n_nonzero_diff": w["n_nonzero_diff"],
                f"{m1}_mean_pinball": float(sub[c1].mean()),
                f"{m2}_mean_pinball": float(sub[c2].mean()),
                f"mean_diff_{m2}_minus_{m1}": float(diff.mean()),
                f"median_diff_{m2}_minus_{m1}": float(np.median(diff)),
                "wilcoxon_statistic": w["wilcoxon_statistic"],
                "wilcoxon_pvalue": w["wilcoxon_pvalue"],
                "interpretation": f"mean_diff_{m2}_minus_{m1} > 0 이면 {m1}가 더 좋음"
            })

    return pd.DataFrame(results)


def save_excel(out_xlsx, comp_pairwise_df, per_tau_pairwise_df, comp_merged_df, tau_merged_df):
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)

    notes = pd.DataFrame({
        "note": [
            "composite_pairwise_wilcoxon: composite pinball loss 기준 세 모델 pairwise Wilcoxon 결과",
            "per_tau_pairwise_wilcoxon: tau별 세 모델 pairwise Wilcoxon 결과",
            "mean_diff_model2_minus_model1 = model2 - model1",
            "diff > 0 이면 model1의 loss가 더 작아서 model1이 더 좋음",
            "loss는 작을수록 좋음",
            "반드시 같은 seed끼리 비교되도록 inner merge 사용"
        ]
    })

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        comp_pairwise_df.to_excel(writer, index=False, sheet_name="composite_pairwise_wilcoxon")
        per_tau_pairwise_df.to_excel(writer, index=False, sheet_name="per_tau_pairwise_wilcoxon")
        comp_merged_df.to_excel(writer, index=False, sheet_name="composite_merged_raw")
        tau_merged_df.to_excel(writer, index=False, sheet_name="per_tau_merged_raw")
        notes.to_excel(writer, index=False, sheet_name="notes")


if __name__ == "__main__":
    # 경로
    mcqf_json_path = r"C:\quantile\grid_result\grid_results_p20_s1\res_s1_p20.json"
    qrf_json_path  = r"C:\quantile\_qrf_result\grid_results_p20_s1_qrf_only\res_s1_p20.json"

    grf_comp_csv_path = r"C:\quantile\grf\fal_result_grf\raw\grf_s1_p20_comp_raw.csv"
    grf_tau_csv_path  = r"C:\quantile\grf\fal_result_grf\raw\grf_s1_p20_per_tau_raw.csv"

    out_xlsx = r"C:\quantile\all_result\excel\pairwise_wilcoxon_mcqf_qrf_grf_s1_p20.xlsx"

    # 로드
    tau, mcqf_comp_df, mcqf_tau_df = load_mcqf_from_json(mcqf_json_path)
    qrf_comp_df, qrf_tau_df = load_qrf_from_json(qrf_json_path, tau_expected=tau)
    grf_comp_df, grf_tau_df = load_grf_raw(
        grf_comp_csv_path,
        grf_tau_csv_path,
        tau_expected=tau
    )

    # composite merge
    comp_merged_df = (
        mcqf_comp_df
        .merge(qrf_comp_df, on="seed", how="inner")
        .merge(grf_comp_df, on="seed", how="inner")
        .sort_values("seed")
        .reset_index(drop=True)
    )

    if comp_merged_df.empty:
        raise ValueError("세 모델의 공통 seed가 없습니다.")

    # per-tau merge
    tau_merged_df = (
        mcqf_tau_df
        .merge(qrf_tau_df, on=["seed", "tau"], how="inner")
        .merge(grf_tau_df, on=["seed", "tau"], how="inner")
        .sort_values(["tau", "seed"])
        .reset_index(drop=True)
    )

    if tau_merged_df.empty:
        raise ValueError("세 모델의 공통 seed-tau 조합이 없습니다.")

    # pairwise Wilcoxon
    comp_pairwise_df = pairwise_wilcoxon_composite(
        all_comp_df=comp_merged_df,
        model_cols={
            "mcqf": "mcqf_comp",
            "qrf": "qrf_comp",
            "grf": "grf_comp",
        }
    )

    per_tau_pairwise_df = pairwise_wilcoxon_per_tau(
        all_tau_df=tau_merged_df,
        pinball_cols={
            "mcqf": "mcqf_pinball",
            "qrf": "qrf_pinball",
            "grf": "grf_pinball",
        }
    )

    # 저장
    save_excel(
        out_xlsx=out_xlsx,
        comp_pairwise_df=comp_pairwise_df,
        per_tau_pairwise_df=per_tau_pairwise_df,
        comp_merged_df=comp_merged_df,
        tau_merged_df=tau_merged_df
    )

    print("[Saved Excel]", out_xlsx)
    print()
    print("[Composite pairwise Wilcoxon]")
    print(comp_pairwise_df.to_string(index=False))
    print()
    print("[Per-tau pairwise Wilcoxon]")
    print(per_tau_pairwise_df.to_string(index=False))