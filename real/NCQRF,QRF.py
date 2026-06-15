import os
import json
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn_quantile import RandomForestQuantileRegressor

from qtrees.forest.qf import QuantileForest

# 설정
OUT_ROOT = "./0611_REALDATA_RESULT_randomsplit_100"
TAU = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

SHARED_ROOT = "./realdata_shared_randomsplit_repeat100"
N_REPEATS = 100
TEST_SIZE = 0.2
INNER_VALID_SIZE = 0.2
BASE_SEED = 100

# 요청하신 4가지 핵심 데이터셋만 유지
DATASETS = {
    "airfoil": {
        "loader": "airfoil_custom",
        "source": "UCI airfoil_self_noise.dat",
        "target_candidates": ["sound"],
        "datetime_parse": False,
    },
    "concrete": {
        "data_id": 4353,
        "source": "OpenML data_id=4353 (Concrete_Data)",
        "target_candidates": [
            "Concrete_compressive_strength",
            "concrete_compressive_strength",
            "strength",
            "target"
        ],
        "datetime_parse": False,
    },
    "insurance_charges": {
        "loader": "insurance_custom",
        "source": "OpenML data_id=46931 (healthcare_insurance_expenses)",
        "target_candidates": ["charges", "expense", "expenses", "target"],
        "datetime_parse": False,
    },
    "yacht_hydrodynamics": {
        "loader": "yacht_custom",
        "source": "OpenML name=yacht_hydrodynamics, version=1",
        "target_candidates": ["Residuary_resistance", "target"],
        "datetime_parse": False,
    }
}

# 유틸리티 함수
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_checkpoint(checkpoint_json):
    if os.path.exists(checkpoint_json):
        with open(checkpoint_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"split_results": []}

def make_split_key(repeat_id):
    return f"repeat{repeat_id}"

def get_done_keys(split_results):
    return {make_split_key(fr["repeat"]) for fr in split_results}

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _safe_xy(X, y):
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
    return X, y

def mean_se(x):
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, se

# 평가지표 (Metrics)
def per_tau_pinball_mcqf(y_true, y_pred_mat, tau):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_mat = np.asarray(y_pred_mat)
    losses = []
    for j, q in enumerate(tau):
        losses.append(mean_pinball_loss(y_true, y_pred_mat[:, j], alpha=float(q)))
    return np.array(losses, dtype=float)

def per_tau_pinball_rfqr(y_true, rfqr_pred, tau):
    y_true = np.asarray(y_true).reshape(-1)
    rfqr_pred = np.asarray(rfqr_pred)
    losses = []
    for j, q in enumerate(tau):
        losses.append(mean_pinball_loss(y_true, rfqr_pred[j, :], alpha=float(q)))
    return np.array(losses, dtype=float)

def composite_from_per_tau(per_tau_losses):
    return float(np.mean(per_tau_losses))

def crossing_percentage_from_preds(pred_mat):
    pred_mat = np.asarray(pred_mat)
    if pred_mat.ndim != 2 or pred_mat.shape[1] < 2:
        return 0.0
    prev = pred_mat[:, :-1]
    nxt = pred_mat[:, 1:]
    crosses = (prev > nxt).astype(np.float32)
    return float(crosses.mean() * 100.0)

def rfqr_pred_to_mat(rfqr_pred):
    return np.asarray(rfqr_pred).T

# 데이터 로드 및 전처리 (핵심 4종 전용)
def load_airfoil_custom(dataset_name="airfoil"):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
    df = pd.read_csv(url, sep="\t", header=None)
    df.columns = ["frequency", "angle", "chord_length", "velocity", "thickness", "sound"]
    y = df["sound"].to_numpy(dtype=np.float64)
    X_df = df.drop(columns=["sound"]).copy()
    meta = {"dataset_name": dataset_name, "source": "UCI airfoil", "n": int(len(y)), "p": int(X_df.shape[1])}
    return X_df.astype(np.float64), y, meta

def load_insurance_custom(dataset_name="insurance_charges"):
    ds = fetch_openml(data_id=46931, as_frame=True, parser="auto")
    df = ds.frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    target_col = "charges"
    y = pd.to_numeric(df[target_col], errors="raise").to_numpy(dtype=np.float64)
    X_df = df.drop(columns=[target_col]).copy()
    cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if len(cat_cols) > 0:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(X_df.median(numeric_only=True))
    meta = {"dataset_name": dataset_name, "source": "OpenML insurance", "n": int(len(y)), "p": int(X_df.shape[1])}
    return X_df.astype(np.float64), y, meta

def load_yacht_custom(dataset_name="yacht_hydrodynamics"):
    ds = fetch_openml(name="yacht_hydrodynamics", version=1, as_frame=True, parser="auto")
    X_df = ds.data.copy().apply(pd.to_numeric, errors="coerce").fillna(ds.data.median(numeric_only=True))
    y = ds.target.astype(np.float64).values
    meta = {"dataset_name": dataset_name, "source": "OpenML yacht", "n": int(len(y)), "p": int(X_df.shape[1])}
    return X_df.astype(np.float64), y, meta

def load_openml_dataset(dataset_name):
    info = DATASETS[dataset_name]
    if info.get("loader") == "airfoil_custom": return load_airfoil_custom(dataset_name)
    if info.get("loader") == "insurance_custom": return load_insurance_custom(dataset_name)
    if info.get("loader") == "yacht_custom": return load_yacht_custom(dataset_name)
    
    ds = fetch_openml(data_id=info["data_id"], as_frame=True, parser="auto")
    df = ds.frame.copy()
    target_col = [c for c in info["target_candidates"] if c in df.columns][0]
    y = pd.to_numeric(df[target_col]).to_numpy(dtype=np.float64)
    X_df = df.drop(columns=[target_col]).copy()
    X_df = pd.get_dummies(X_df, drop_first=False).apply(pd.to_numeric, errors="coerce").fillna(X_df.median(numeric_only=True))
    meta = {"dataset_name": dataset_name, "source": info["source"], "n": int(len(y)), "p": int(X_df.shape[1])}
    return X_df.astype(np.float64), y, meta

# Shared Data 관리
def get_shared_paths(dataset_name, shared_root=SHARED_ROOT):
    dataset_dir = os.path.join(shared_root, dataset_name)
    return {
        "dir": dataset_dir,
        "data_csv": os.path.join(dataset_dir, f"{dataset_name}_full_data.csv"),
        "split_json": os.path.join(dataset_dir, f"{dataset_name}_splits_random.json"),
        "meta_json": os.path.join(dataset_dir, f"{dataset_name}_meta_random.json"),
    }

def prepare_shared_dataset_if_needed(dataset_name, shared_root=SHARED_ROOT, n_repeats=N_REPEATS, test_size=TEST_SIZE, inner_valid_size=INNER_VALID_SIZE, base_seed=BASE_SEED):
    paths = get_shared_paths(dataset_name, shared_root)
    if all(os.path.exists(paths[k]) for k in ["data_csv", "split_json", "meta_json"]):
        return

    X_df, y, meta = load_openml_dataset(dataset_name)
    n = len(y)
    row_id = np.arange(1, n + 1, dtype=int)
    full_df = X_df.copy()
    full_df.insert(0, "row_id", row_id)
    full_df["y"] = y
    ensure_dir(paths["dir"])
    full_df.to_csv(paths["data_csv"], index=False)

    split_json = {"dataset_name": dataset_name, "splits": []}
    all_idx = np.arange(n)
    for repeat_id in range(1, n_repeats + 1):
        repeat_seed = base_seed + repeat_id * 1000
        train_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=repeat_seed, shuffle=True)
        in_tr_idx, in_va_idx = train_test_split(train_idx, test_size=inner_valid_size, random_state=repeat_seed + 1, shuffle=True)
        split_json["splits"].append({
            "repeat": int(repeat_id), "repeat_seed": int(repeat_seed),
            "train_row_id": row_id[train_idx].tolist(), "test_row_id": row_id[test_idx].tolist(),
            "inner_train_row_id": row_id[in_tr_idx].tolist(), "inner_valid_row_id": row_id[in_va_idx].tolist()
        })
    save_json(split_json, paths["split_json"])
    save_json(meta, paths["meta_json"])

def load_shared_dataset_and_splits(dataset_name, shared_root=SHARED_ROOT):
    paths = get_shared_paths(dataset_name, shared_root)
    full_df = pd.read_csv(paths["data_csv"])
    with open(paths["split_json"], "r") as f: split_info = json.load(f)
    with open(paths["meta_json"], "r") as f: meta = json.load(f)
    return full_df, split_info, meta

# Grid Search
def get_grid_params(p):
    return list(ParameterGrid({
        "n_estimators": [1000],
        "max_depth": [5, 8, 10, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": sorted(set([int(np.sqrt(p)), p]))
    }))

def grid_search_mcqf(Xtr, ytr, Xva, yva, tau, seed):
    Xtr, ytr = _safe_xy(Xtr, ytr)
    Xva, yva = _safe_xy(Xva, yva)
    best_score, best_params = np.inf, None
    for params in get_grid_params(Xtr.shape[1]):
        model = QuantileForest(**params, tau=tau, n_jobs=-1, random_state=seed, bootstrap=False, max_samples=0.632)
        model.fit(Xtr, ytr)
        score = composite_from_per_tau(per_tau_pinball_mcqf(yva, model.predict(Xva), tau))
        if score < best_score: best_score, best_params = score, params
    return best_params, float(best_score)

def grid_search_rfqr(Xtr, ytr, Xva, yva, tau, seed):
    Xtr, ytr = _safe_xy(Xtr, ytr)
    Xva, yva = _safe_xy(Xva, yva)
    best_score, best_params = np.inf, None
    for params in get_grid_params(Xtr.shape[1]):
        model = RandomForestQuantileRegressor(**params, q=tau, n_jobs=-1, random_state=seed)
        model.fit(Xtr, ytr)
        score = composite_from_per_tau(per_tau_pinball_rfqr(yva, model.predict(Xva), tau))
        if score < best_score: best_score, best_params = score, params
    return best_params, float(best_score)

# 실험 실행
def run_repeated_randomsplit_dataset(dataset_name):
    prepare_shared_dataset_if_needed(dataset_name)
    full_df, split_info, meta = load_shared_dataset_and_splits(dataset_name)
    
    dataset_out = os.path.join(OUT_ROOT, dataset_name)
    raw_dir = os.path.join(dataset_out, "raw")
    ensure_dir(raw_dir)
    
    checkpoint_json = os.path.join(raw_dir, f"checkpoint_{dataset_name}.json")
    checkpoint = load_checkpoint(checkpoint_json)
    split_results = checkpoint.get("split_results", [])
    done_keys = get_done_keys(split_results)

    feature_cols = [c for c in full_df.columns if c not in ["row_id", "y"]]
    rowid_to_pos = {int(rid): i for i, rid in enumerate(full_df["row_id"])}

    for split_one in split_info["splits"]:
        repeat_id = split_one["repeat"]
        if make_split_key(repeat_id) in done_keys: continue

        print(f"[{dataset_name}] Repeat {repeat_id}/{N_REPEATS}")
        
        # 데이터 분할 로직 (Row ID 기반)
        def get_sub(rids): return full_df.iloc[[rowid_to_pos[rid] for rid in rids]]
        df_tr, df_te = get_sub(split_one["train_row_id"]), get_sub(split_one["test_row_id"])
        df_in_tr, df_in_va = get_sub(split_one["inner_train_row_id"]), get_sub(split_one["inner_valid_row_id"])

        X_in_tr, y_in_tr = df_in_tr[feature_cols].values, df_in_tr["y"].values
        X_in_va, y_in_va = df_in_va[feature_cols].values, df_in_va["y"].values
        X_tr, y_tr = df_tr[feature_cols].values, df_tr["y"].values
        X_te, y_te = df_te[feature_cols].values, df_te["y"].values

        # 학습 및 평가
        b_m_p, b_m_v = grid_search_mcqf(X_in_tr, y_in_tr, X_in_va, y_in_va, TAU, 1000+repeat_id)
        mcqf = QuantileForest(**b_m_p, tau=TAU, n_jobs=-1, random_state=3000+repeat_id, bootstrap=False).fit(X_tr, y_tr)
        m_pred = mcqf.predict(X_te)

        b_r_p, b_r_v = grid_search_rfqr(X_in_tr, y_in_tr, X_in_va, y_in_va, TAU, 2000+repeat_id)
        rfqr = RandomForestQuantileRegressor(**b_r_p, q=TAU, n_jobs=-1, random_state=4000+repeat_id).fit(X_tr, y_tr)
        r_pred = rfqr.predict(X_te)

        split_results.append({
            "repeat": repeat_id,
            "test": {
                "mcqf_comp": float(composite_from_per_tau(per_tau_pinball_mcqf(y_te, m_pred, TAU))),
                "rfqr_comp": float(composite_from_per_tau(per_tau_pinball_rfqr(y_te, r_pred, TAU))),
                "mcqf_cross_pct": crossing_percentage_from_preds(m_pred),
                "rfqr_cross_pct": crossing_percentage_from_preds(rfqr_pred_to_mat(r_pred))
            }
        })
        save_json({"split_results": split_results}, checkpoint_json)

    print(f"Final Summary for {dataset_name} saved in {checkpoint_json}")

if __name__ == "__main__":
    for ds in DATASETS.keys():
        run_repeated_randomsplit_dataset(ds)
