import os
import json
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import mean_pinball_loss
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal

warnings.filterwarnings("ignore")

# ============================================================
# [1] 설정
# ============================================================
OUT_ROOT = "./0511_REALDATA_RESULT_randomsplit_100"
SHARED_ROOT = "./realdata_shared_randomsplit_repeat100"
TAU = np.arange(0.1, 1.0, 0.2)
N_REPEATS = 100
BASE_SEED = 100

# ============================================================
# [2] 공통 유틸리티
# ============================================================
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return {"split_results": []}

def subset_by_row_id(full_df, row_ids):
    df = full_df[full_df["row_id"].isin(row_ids)].set_index("row_id").loc[row_ids].reset_index()
    X = df.drop(columns=["row_id", "y"]).values
    y = df["y"].values
    return np.ascontiguousarray(X, dtype=np.float64), y.reshape(-1)

def get_metrics(y_true, pred_mat):
    losses = [mean_pinball_loss(y_true, pred_mat[:, j], alpha=q) for j, q in enumerate(TAU)]
    cross = (pred_mat[:, :-1] > pred_mat[:, 1:]).mean() * 100
    return losses, np.mean(losses), cross

# ============================================================
# [3] 모델별 Optuna 파라미터 제안 로직
# ============================================================
def suggest_tree_params(model_name, trial, p):
    if model_name == "lgbm":
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [0.025, 0.05, 0.1, 0.2, 0.3]),
            "num_leaves": trial.suggest_categorical("num_leaves", [3, 7, 15, 31, 127]),
            "top_rate": trial.suggest_float("top_rate", 0.1, 0.5),
            "other_rate": trial.suggest_float("other_rate", 0.05, 0.2),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 70),
        }
    elif model_name == "xgb":
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [0.025, 0.05, 0.1, 0.2, 0.3]),
            "max_depth": trial.suggest_categorical("max_depth", [2, 3, 5, 7, 10]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 2.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 70),
        }
    elif model_name == "ngboost":
        max_feat_vals = sorted(set([int(np.sqrt(p)), int(0.25*p), p]))
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [0.025, 0.05, 0.1, 0.2, 0.3]),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 70),
            "max_features": trial.suggest_categorical("max_features", [v for v in max_feat_vals if v >= 1]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

# ============================================================
# [4] 모델 학습 및 예측 라우터
# ============================================================
def fit_predict_model(model_name, Xtr, ytr, Xva, yva, Xfit, yfit, Xte, n_trials, seed):
    p_dim = Xtr.shape[1]

    def objective(trial):
        params = suggest_tree_params(model_name, trial, p_dim)
        preds = []
        for q in TAU:
            if model_name == "lgbm":
                model = LGBMRegressor(objective="quantile", alpha=q, n_estimators=500, random_state=seed, verbose=-1, boosting_type="goss", **params)
            elif model_name == "xgb":
                model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, n_estimators=500, random_state=seed, tree_method="hist", **params)
            elif model_name == "ngboost":
                base = DecisionTreeRegressor(max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"], random_state=seed)
                model = NGBRegressor(Dist=Normal, Base=base, n_estimators=500, learning_rate=params["learning_rate"], minibatch_frac=params["subsample"], verbose=False, random_state=seed)
            
            model.fit(Xtr, ytr)
            preds.append(model.predict(Xva) if model_name != "ngboost" else model.pred_dist(Xva).ppf(q))
        
        return np.mean([mean_pinball_loss(yva, preds[i], alpha=q) for i, q in enumerate(TAU)])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    # 최종 예측
    final_preds = []
    for q in TAU:
        if model_name == "lgbm":
            m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=1000, random_state=seed, verbose=-1, boosting_type="goss", **best)
        elif model_name == "xgb":
            m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, n_estimators=1000, random_state=seed, tree_method="hist", **best)
        elif model_name == "ngboost":
            base = DecisionTreeRegressor(max_depth=best["max_depth"], min_samples_leaf=best["min_samples_leaf"], max_features=best["max_features"], random_state=seed)
            m = NGBRegressor(Dist=Normal, Base=base, n_estimators=1000, learning_rate=best["learning_rate"], minibatch_frac=best["subsample"], verbose=False, random_state=seed)
        
        m.fit(Xfit, yfit)
        final_preds.append(m.predict(Xte) if model_name != "ngboost" else m.pred_dist(Xte).ppf(q))
    
    return np.column_stack(final_preds), best, study.best_value

# ============================================================
# [5] 메인 실험 루프
# ============================================================
def run_experiment(dataset_name, model_name, n_trials):
    print(f"\n{'='*60}\n[START] Dataset: {dataset_name} | Model: {model_name}\n{'='*60}")
    
    # 데이터 로드 (원본 공유 데이터 경로)
    data_path = os.path.join(SHARED_ROOT, dataset_name)
    full_df = pd.read_csv(os.path.join(data_path, f"{dataset_name}_full_data.csv"))
    with open(os.path.join(data_path, f"{dataset_name}_splits_random.json")) as f: split_info = json.load(f)
    
    out_dir = os.path.join(OUT_ROOT, model_name, dataset_name)
    checkpoint_path = os.path.join(out_dir, "result.json")
    res = load_checkpoint(checkpoint_path)
    done_keys = {f"repeat{r['repeat']}" for r in res.get("split_results", [])}

    for sp in split_info["splits"]:
        repeat_id = sp["repeat"]
        if f"repeat{repeat_id}" in done_keys: continue

        Xfit, yfit = subset_by_row_id(full_df, sp["train_row_id"])
        Xte, yte = subset_by_row_id(full_df, sp["test_row_id"])
        Xtr, ytr = subset_by_row_id(full_df, sp["inner_train_row_id"])
        Xva, yva = subset_by_row_id(full_df, sp["inner_valid_row_id"])

        pred_mat, best_p, b_val = fit_predict_model(model_name, Xtr, ytr, Xva, yva, Xfit, yfit, Xte, n_trials, BASE_SEED+repeat_id)
        
        losses, avg_loss, cross = get_metrics(yte, pred_mat)
        res["split_results"].append({
            "repeat": repeat_id,
            "best_params": best_p,
            "test_composite": float(avg_loss),
            "test_crossing": float(cross)
        })
        
        save_json(res, checkpoint_path)
        print(f" Repeat {repeat_id}/{N_REPEATS} | Loss: {avg_loss:.6f} | Crossing: {cross:.3f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["airfoil", "insurance_charges", "yacht_hydrodynamics"])
    parser.add_argument("--models", nargs="+", default=["lgbm", "xgb", "ngboost"])
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    for ds in args.datasets:
        for md in args.models:
            run_experiment(ds, md, args.n_trials)
