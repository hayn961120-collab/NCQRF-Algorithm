import sys
sys.path.append(r"C:\quantile")

import os
import json
import numpy as np
from sklearn.metrics import mean_pinball_loss

from qrf_sim.simul_utils_n1000 import gen_simul1, gen_simul2, gen_simul4

def composite_pinball(y_true, y_pred_mat, tau):
    y_true = np.asarray(y_true).reshape(-1)
    losses = []
    for j, q in enumerate(tau):
        losses.append(mean_pinball_loss(y_true, y_pred_mat[:, j], alpha=q))
    return float(np.mean(losses)), np.array(losses, dtype=float)


def get_scenario_fn(scenario_id):
    if scenario_id == 1:
        return gen_simul1
    elif scenario_id == 2:
        return gen_simul2
    elif scenario_id == 4:
        return gen_simul4
    else:
        raise ValueError("scenario_id must be in {1,2,4}")


def make_data(scenario_id, train_n, seed, p):
    """
    - Scenario 1: gen_simul1 (GEV, heteroscedastic, c = -1/2)
    - Scenario 2: gen_simul2 (Normal, heteroscedastic)
    - Scenario 4: gen_simul4 (GRFpaper, 0기준으로 왼쪽 분산1, 오른쪽 분산2)
    """
    fn = get_scenario_fn(scenario_id)

    if scenario_id == 1:
        train, valid, test = fn(train_n=train_n, scale=0.1, seed=seed, input_dim=p)
    elif scenario_id == 2:
        train, valid, test = fn(train_n=train_n, scale=0.1, seed=seed, input_dim=p)
    elif scenario_id == 4:
        train, valid, test = fn(train_n = train_n, seed = seed, input_dim=p)

    return train, valid, test


def init_empty_result(tau):
    return {
        "model": {"comp": [], "per_tau": []},
        "best_params": [],
        "best_val": [],
        "tau": np.array(tau, dtype=float),
    }


def load_res_json(path):
    with open(path, "r", encoding="utf-8") as f:
        res = json.load(f)

    res["model"]["comp"] = np.array(res["model"]["comp"], dtype=float)
    res["model"]["per_tau"] = np.array(res["model"]["per_tau"], dtype=float)
    res["tau"] = np.array(res["tau"], dtype=float)
    return res


def add_summary_to_res(res):
    tau = res["tau"]
    per_tau = res["model"]["per_tau"]
    comp = res["model"]["comp"]

    mean_tau = per_tau.mean(axis=0)
    std_tau = per_tau.std(axis=0, ddof=1) if len(per_tau) > 1 else np.zeros_like(mean_tau)
    comp_mean = float(comp.mean())
    comp_std = float(comp.std(ddof=1)) if len(comp) > 1 else 0.0

    res["summary"] = {
        "tau": tau.tolist(),
        "mean_tau": mean_tau.tolist(),
        "std_tau": std_tau.tolist(),
        "comp_mean": comp_mean,
        "comp_std": comp_std,
    }
    return res


def summarize_result(res, model_name):
    tau = res["tau"]
    per_tau = res["model"]["per_tau"]
    comp = res["model"]["comp"]

    mean_tau = per_tau.mean(axis=0)
    std_tau = per_tau.std(axis=0, ddof=1) if len(per_tau) > 1 else np.zeros_like(mean_tau)
    comp_mean = float(comp.mean())
    comp_std = float(comp.std(ddof=1)) if len(comp) > 1 else 0.0

    print("\n================ Summary ================")
    print("model:", model_name)
    print("tau:", tau)
    print("\n[Per-tau pinball loss on TEST]")
    print("mean:", mean_tau)
    print("std :", std_tau)
    print("\n[Composite mean pinball on TEST]")
    print("comp mean:", comp_mean, "std:", comp_std)
    print("=========================================\n")


def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def save_res_json(res, out_dir, scenario_id, p, model_tag):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"res_{model_tag}_s{scenario_id}_p{p}.json")
    tmp_path = path + ".tmp"

    dumpable = _to_jsonable(res)

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(dumpable, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)
    return path