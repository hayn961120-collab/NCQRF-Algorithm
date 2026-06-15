import sys
sys.path.append("/home/hayn08/0505quantile")

import os
import json
from collections import Counter

import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import ParameterGrid

from qtrees.forest.qf import QuantileForest
from qrf_sim.simul_utils_0313 import gen_simul1, gen_simul2, gen_simul3, gen_simul4


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
    elif scenario_id == 3:
        return gen_simul3
    elif scenario_id == 4:
        return gen_simul4
    else:
        raise ValueError("scenario_id must be in {1,2,3,4}")


def make_data(scenario_id, train_n, seed, p):
    fn = get_scenario_fn(scenario_id)

    if scenario_id in [1, 2, 3]:
        train, valid, test = fn(train_n=train_n, scale=0.1, seed=seed, input_dim=p)
    elif scenario_id == 4:
        train, valid, test = fn(train_n=train_n, seed=seed, input_dim=p)

    return train, valid, test


def get_qtree_split_usage(model, signal_idx=0):
    root_features = []
    all_features = []

    for est in model.estimators_:
        tree = est.tree_

        root_f = tree.feature[0]
        if root_f >= 0:
            root_features.append(int(root_f))

        fs = tree.feature
        fs = fs[fs >= 0]
        all_features.extend([int(x) for x in fs.tolist()])

    root_counts = Counter(root_features)
    all_counts = Counter(all_features)

    n_root = len(root_features)
    n_all = len(all_features)

    return {
        "root_counts": dict(root_counts),
        "all_counts": dict(all_counts),
        "root_x1_rate": root_counts.get(signal_idx, 0) / n_root if n_root > 0 else np.nan,
        "all_x1_rate": all_counts.get(signal_idx, 0) / n_all if n_all > 0 else np.nan,
        "n_root": n_root,
        "n_all_splits": n_all,
    }


def get_qtree_tree_structure(model):
    n_splits = []
    n_leaves = []
    max_depths = []
    mean_leaf_sizes = []

    for est in model.estimators_:
        tree = est.tree_

        split_count = int(np.sum(tree.feature >= 0))
        leaf_mask = tree.feature < 0
        leaf_count = int(np.sum(leaf_mask))
        depth = int(tree.max_depth)

        leaf_sizes = tree.n_node_samples[leaf_mask]
        mean_leaf_size = float(np.mean(leaf_sizes)) if leaf_count > 0 else np.nan

        n_splits.append(split_count)
        n_leaves.append(leaf_count)
        max_depths.append(depth)
        mean_leaf_sizes.append(mean_leaf_size)

    return {
        "mean_splits_per_tree": float(np.mean(n_splits)),
        "mean_leaves_per_tree": float(np.mean(n_leaves)),
        "mean_depth_per_tree": float(np.mean(max_depths)),
        "mean_leaf_size": float(np.mean(mean_leaf_sizes)),
    }

def get_sklearn_child_size_stats(model):
    min_child_ratios = []
    left_ratios = []
    right_ratios = []
    parent_sizes = []
    left_sizes = []
    right_sizes = []

    for est in model.estimators_:
        tree = est.tree_

        for node_id in range(tree.node_count):
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]

            # leaf는 children_left/right가 -1
            if left == -1 or right == -1:
                continue

            parent_n = tree.n_node_samples[node_id]
            left_n = tree.n_node_samples[left]
            right_n = tree.n_node_samples[right]

            if parent_n <= 0:
                continue

            left_ratio = left_n / parent_n
            right_ratio = right_n / parent_n
            min_ratio = min(left_ratio, right_ratio)

            parent_sizes.append(parent_n)
            left_sizes.append(left_n)
            right_sizes.append(right_n)
            left_ratios.append(left_ratio)
            right_ratios.append(right_ratio)
            min_child_ratios.append(min_ratio)

    min_child_ratios = np.array(min_child_ratios, dtype=float)

    return {
        "mean_min_child_ratio": float(np.nanmean(min_child_ratios)),
        "median_min_child_ratio": float(np.nanmedian(min_child_ratios)),
        "p05_min_child_ratio": float(np.nanpercentile(min_child_ratios, 5)),
        "p10_min_child_ratio": float(np.nanpercentile(min_child_ratios, 10)),
        "p25_min_child_ratio": float(np.nanpercentile(min_child_ratios, 25)),
        "extreme_005_rate": float(np.mean(min_child_ratios < 0.05)),
        "extreme_010_rate": float(np.mean(min_child_ratios < 0.10)),
        "n_internal_splits_for_balance": int(len(min_child_ratios)),
    }
def load_res_json(path):
    with open(path, "r", encoding="utf-8") as f:
        res = json.load(f)

    res["mcqf"]["comp"] = np.array(res["mcqf"]["comp"], dtype=float)
    res["mcqf"]["per_tau"] = np.array(res["mcqf"]["per_tau"], dtype=float)
    res["tau"] = np.array(res["tau"], dtype=float)

    if "split_usage" not in res:
        res["split_usage"] = {
            "root_x1_rate": [],
            "all_x1_rate": [],
            "root_counts": [],
            "all_counts": [],
            "n_root": [],
            "n_all_splits": [],
        }

    if "tree_structure" not in res:
        res["tree_structure"] = {
            "mean_splits_per_tree": [],
            "mean_leaves_per_tree": [],
            "mean_depth_per_tree": [],
            "mean_leaf_size": [],
        }
    if "child_balance" not in res:
        res["child_balance"] = {
            "mean_min_child_ratio": [],
            "median_min_child_ratio": [],
            "p05_min_child_ratio": [],
            "p10_min_child_ratio": [],
            "p25_min_child_ratio": [],
            "extreme_005_rate": [],
            "extreme_010_rate": [],
            "n_internal_splits_for_balance": [],
        }
    return res


def init_empty_result(tau):
    return {
        "mcqf": {"comp": [], "per_tau": []},
        "best_params": {"mcqf": []},
        "best_val": {"mcqf": []},
        "split_usage": {
            "root_x1_rate": [],
            "all_x1_rate": [],
            "root_counts": [],
            "all_counts": [],
            "n_root": [],
            "n_all_splits": [],
        },
        "tree_structure": {
            "mean_splits_per_tree": [],
            "mean_leaves_per_tree": [],
            "mean_depth_per_tree": [],
            "mean_leaf_size": [],
        },
        "tau": np.array(tau, dtype=float),
        "child_balance": {
            "mean_min_child_ratio": [],
            "median_min_child_ratio": [],
            "p05_min_child_ratio": [],
            "p10_min_child_ratio": [],
            "p25_min_child_ratio": [],
            "extreme_005_rate": [],
            "extreme_010_rate": [],
            "n_internal_splits_for_balance": [],
        },
    }


def print_running_summary(res, rep_idx):
    tau = np.array(res["tau"], dtype=float)
    mcqf_per = np.array(res["mcqf"]["per_tau"], dtype=float)
    mcqf_comp = np.array(res["mcqf"]["comp"], dtype=float)

    n_done = len(mcqf_comp)

    mcqf_mean_tau = mcqf_per.mean(axis=0)
    mcqf_std_tau = mcqf_per.std(axis=0, ddof=1) if n_done > 1 else np.zeros_like(mcqf_mean_tau)

    mcqf_comp_mean = float(mcqf_comp.mean())
    mcqf_comp_std = float(mcqf_comp.std(ddof=1)) if n_done > 1 else 0.0

    root_x1_mean = np.nanmean(res["split_usage"]["root_x1_rate"])
    all_x1_mean = np.nanmean(res["split_usage"]["all_x1_rate"])

    mean_splits = np.nanmean(res["tree_structure"]["mean_splits_per_tree"])
    mean_leaves = np.nanmean(res["tree_structure"]["mean_leaves_per_tree"])
    mean_depth = np.nanmean(res["tree_structure"]["mean_depth_per_tree"])
    mean_leaf_size = np.nanmean(res["tree_structure"]["mean_leaf_size"])

    print(f"\n  [Running summary: 0 ~ {rep_idx} / n={n_done}]")
    print("    tau:", tau)
    print("    MCQF mean per-tau:", np.round(mcqf_mean_tau, 6))
    print(f"    MCQF comp mean={mcqf_comp_mean:.6f}, std={mcqf_comp_std:.6f}")
    print(f"    root X1 rate mean={root_x1_mean:.4f}")
    print(f"    all split X1 rate mean={all_x1_mean:.4f}")
    print(f"    mean splits/tree={mean_splits:.4f}")
    print(f"    mean leaves/tree={mean_leaves:.4f}")
    print(f"    mean depth/tree={mean_depth:.4f}")
    print(f"    mean leaf size={mean_leaf_size:.4f}")


def make_max_features_grid(p):
    if p == 1:
        return [1]
    return [p]


def make_min_samples_leaf_grid(train_n):
    return [30]


def get_grid_params(p, train_n):
    param_grid = {
        "n_estimators": [1000],
        "max_depth": [None],
        "min_samples_leaf": make_min_samples_leaf_grid(train_n),
        "max_features": make_max_features_grid(p),
    }
    return list(ParameterGrid(param_grid))


def grid_search_mcqf(Xtr, ytr, Xva, yva, tau, seed, p, train_n):
    grid = get_grid_params(p, train_n)

    best_score = np.inf
    best_params = None

    for params in grid:
        model = QuantileForest(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            tau=tau,
            n_jobs=4,
            oob_score=False,
            random_state=seed,
            bootstrap=False,
            max_samples=0.632,
        )
        model.fit(Xtr, ytr)

        pred = model.predict(Xva)
        comp, _ = composite_pinball(yva, pred, tau)

        if comp < best_score:
            best_score = comp
            best_params = params.copy()

    return best_params, float(best_score)


def run_one_setting(
    scenario_id=1,
    p=20,
    train_n=1000,
    mc_repeats=100,
    tau=np.arange(0.1, 1.0, 0.2),
    out_dir="./0602result/grid_results_p20_s1_n1000_tree_structure",
):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"res_s{scenario_id}_p{p}.json")

    if os.path.exists(save_path):
        print(f"[Resume] loading existing result: {save_path}")
        out = load_res_json(save_path)

        out["mcqf"]["comp"] = list(out["mcqf"]["comp"])
        out["mcqf"]["per_tau"] = [np.array(x, dtype=float) for x in out["mcqf"]["per_tau"]]
    else:
        out = init_empty_result(tau)

    start_rep = len(out["mcqf"]["comp"])
    print(f"[Progress] completed repetitions: {start_rep}/{mc_repeats}")

    K = len(tau)

    for i in range(start_rep, mc_repeats):
        print(f"[scenario={scenario_id} | p={p}] repetition {i}/{mc_repeats-1}")

        train, valid, test = make_data(scenario_id, train_n=train_n, seed=i, p=p)

        Xtr = train["data"]
        ytr = np.asarray(train["label"]).reshape(-1)
        Xva = valid["data"]
        yva = np.asarray(valid["label"]).reshape(-1)
        Xte = test["data"]
        yte = np.asarray(test["label"]).reshape(-1)

        best_mcqf_params, best_mcqf_val = grid_search_mcqf(
            Xtr, ytr, Xva, yva, tau, seed=i, p=p, train_n=train_n
        )

        mcqf = QuantileForest(
            n_estimators=best_mcqf_params["n_estimators"],
            max_depth=best_mcqf_params["max_depth"],
            min_samples_leaf=best_mcqf_params["min_samples_leaf"],
            max_features=best_mcqf_params["max_features"],
            tau=tau,
            n_jobs=4,
            oob_score=False,
            random_state=i,
            bootstrap=False,
            max_samples=0.632,
        )
        mcqf.fit(Xtr, ytr)
        balance = get_sklearn_child_size_stats(mcqf)
        
        usage = get_qtree_split_usage(mcqf, signal_idx=0)
        tree_struct = get_qtree_tree_structure(mcqf)

        mcqf_pred = mcqf.predict(Xte)
        mcqf_comp, mcqf_per_tau = composite_pinball(yte, mcqf_pred, tau)

        out["mcqf"]["comp"].append(float(mcqf_comp))
        out["mcqf"]["per_tau"].append(np.array(mcqf_per_tau, dtype=float))
        out["best_params"]["mcqf"].append(best_mcqf_params)
        out["best_val"]["mcqf"].append(float(best_mcqf_val))

        out["split_usage"]["root_x1_rate"].append(float(usage["root_x1_rate"]))
        out["split_usage"]["all_x1_rate"].append(float(usage["all_x1_rate"]))
        out["split_usage"]["root_counts"].append(usage["root_counts"])
        out["split_usage"]["all_counts"].append(usage["all_counts"])
        out["split_usage"]["n_root"].append(int(usage["n_root"]))
        out["split_usage"]["n_all_splits"].append(int(usage["n_all_splits"]))

        for k, v in tree_struct.items():
            out["tree_structure"][k].append(float(v))

        for k, v in balance.items():
            out["child_balance"][k].append(float(v))
        
        print(
            f"  MCQF: val(best)={best_mcqf_val:.6f} "
            f"test(comp)={mcqf_comp:.6f} "
            f"root_x1={usage['root_x1_rate']:.4f} "
            f"all_x1={usage['all_x1_rate']:.4f} "
            f"splits/tree={tree_struct['mean_splits_per_tree']:.2f} "
            f"leaves/tree={tree_struct['mean_leaves_per_tree']:.2f} "
            f"depth={tree_struct['mean_depth_per_tree']:.2f} "
            f"leaf_size={tree_struct['mean_leaf_size']:.2f} "
            f"best={best_mcqf_params}"
        )

        temp_out = {
            "mcqf": {
                "comp": np.array(out["mcqf"]["comp"], dtype=float),
                "per_tau": np.vstack(out["mcqf"]["per_tau"])
                if len(out["mcqf"]["per_tau"]) > 0
                else np.empty((0, K)),
            },
            "best_params": out["best_params"],
            "best_val": out["best_val"],
            "split_usage": out["split_usage"],
            "tree_structure": out["tree_structure"],
            "child_balance": out["child_balance"],  # 이 줄 추가
            "tau": np.array(tau, dtype=float),
        }

        temp_out = add_summary_to_res(temp_out)
        print_running_summary(temp_out, i)
        save_res_json(temp_out, out_dir, scenario_id, p)
        print(f"  [Checkpoint saved] repetition {i} complete")

    out["mcqf"]["comp"] = np.array(out["mcqf"]["comp"], dtype=float)
    out["mcqf"]["per_tau"] = (
        np.vstack(out["mcqf"]["per_tau"])
        if len(out["mcqf"]["per_tau"]) > 0
        else np.empty((0, K))
    )
    out["tau"] = np.array(tau, dtype=float)

    out = add_summary_to_res(out)
    save_res_json(out, out_dir, scenario_id, p)

    return out


def summarize_result(res):
    tau = res["tau"]
    mcqf_per = res["mcqf"]["per_tau"]
    mcqf_comp = res["mcqf"]["comp"]

    print("\n================ Summary ================")
    print("tau:", tau)
    print("\n[Per-tau pinball loss on TEST]")
    print("MCQF mean:", mcqf_per.mean(axis=0))
    print("MCQF std :", mcqf_per.std(axis=0, ddof=1))
    print("\n[Composite mean pinball on TEST]")
    print("MCQF comp mean:", float(mcqf_comp.mean()), "std:", float(mcqf_comp.std(ddof=1)))
    print("\n[Split usage]")
    print("Root X1 rate mean:", float(np.nanmean(res["split_usage"]["root_x1_rate"])))
    print("All split X1 rate mean:", float(np.nanmean(res["split_usage"]["all_x1_rate"])))
    print("\n[Tree structure]")
    print("Mean splits/tree:", float(np.nanmean(res["tree_structure"]["mean_splits_per_tree"])))
    print("Mean leaves/tree:", float(np.nanmean(res["tree_structure"]["mean_leaves_per_tree"])))
    print("Mean depth/tree:", float(np.nanmean(res["tree_structure"]["mean_depth_per_tree"])))
    print("Mean leaf size:", float(np.nanmean(res["tree_structure"]["mean_leaf_size"])))
    print("=========================================\n")
    print("\n[Child balance]")
    print("Mean min child ratio:", float(np.nanmean(res["child_balance"]["mean_min_child_ratio"])))
    print("Median min child ratio:", float(np.nanmean(res["child_balance"]["median_min_child_ratio"])))
    print("P05 min child ratio:", float(np.nanmean(res["child_balance"]["p05_min_child_ratio"])))
    print("P10 min child ratio:", float(np.nanmean(res["child_balance"]["p10_min_child_ratio"])))
    print("Extreme <0.05 rate:", float(np.nanmean(res["child_balance"]["extreme_005_rate"])))
    print("Extreme <0.10 rate:", float(np.nanmean(res["child_balance"]["extreme_010_rate"])))


def add_summary_to_res(res):
    tau = res["tau"]
    mcqf_per = np.array(res["mcqf"]["per_tau"], dtype=float)
    mcqf_comp = np.array(res["mcqf"]["comp"], dtype=float)

    res["summary"] = {
        "tau": tau.tolist(),
        "mcqf_mean_tau": mcqf_per.mean(axis=0).tolist(),
        "mcqf_std_tau": mcqf_per.std(axis=0, ddof=1).tolist(),
        "mcqf_comp_mean": float(mcqf_comp.mean()),
        "mcqf_comp_std": float(mcqf_comp.std(ddof=1)),
        "root_x1_mean": float(np.nanmean(res["split_usage"]["root_x1_rate"])),
        "root_x1_std": float(np.nanstd(res["split_usage"]["root_x1_rate"], ddof=1)),
        "all_x1_mean": float(np.nanmean(res["split_usage"]["all_x1_rate"])),
        "all_x1_std": float(np.nanstd(res["split_usage"]["all_x1_rate"], ddof=1)),
        "mean_splits_per_tree": float(np.nanmean(res["tree_structure"]["mean_splits_per_tree"])),
        "mean_leaves_per_tree": float(np.nanmean(res["tree_structure"]["mean_leaves_per_tree"])),
        "mean_depth_per_tree": float(np.nanmean(res["tree_structure"]["mean_depth_per_tree"])),
        "mean_leaf_size": float(np.nanmean(res["tree_structure"]["mean_leaf_size"])),
    }
    res["summary"]["mean_min_child_ratio"] = float(np.nanmean(res["child_balance"]["mean_min_child_ratio"]))
    res["summary"]["median_min_child_ratio"] = float(np.nanmean(res["child_balance"]["median_min_child_ratio"]))
    res["summary"]["p05_min_child_ratio"] = float(np.nanmean(res["child_balance"]["p05_min_child_ratio"]))
    res["summary"]["p10_min_child_ratio"] = float(np.nanmean(res["child_balance"]["p10_min_child_ratio"]))
    res["summary"]["p25_min_child_ratio"] = float(np.nanmean(res["child_balance"]["p25_min_child_ratio"]))
    res["summary"]["extreme_005_rate"] = float(np.nanmean(res["child_balance"]["extreme_005_rate"]))
    res["summary"]["extreme_010_rate"] = float(np.nanmean(res["child_balance"]["extreme_010_rate"]))
        
    return res


def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def save_res_json(res, out_dir, scenario_id, p):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"res_s{scenario_id}_p{p}.json")
    tmp_path = path + ".tmp"

    dumpable = _to_jsonable(res)

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(dumpable, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)
    return path


if __name__ == "__main__":
    tau = [0.05, 0.1, 0.9, 0.95]

    train_n = 1000
    mc_repeats = 100
    scenarios = [1]
    ps = [1]

    out_dir = "./0610_ncqrf_p1_s1_n1000_tree_structurea"

    for p in ps:
        for scenario_id in scenarios:
            res = run_one_setting(
                scenario_id=scenario_id,
                p=p,
                train_n=train_n,
                mc_repeats=mc_repeats,
                tau=tau,
                out_dir=out_dir,
            )
            summarize_result(res)
            path = save_res_json(res, out_dir=out_dir, scenario_id=scenario_id, p=p)
            print("[Saved]", path)