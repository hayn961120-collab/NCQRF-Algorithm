# run_realdata_mqboost_lgbm.py
# Real-data MQBoost-LGBM runner based on the random-split real-data LGBM code structure.

import os
import json
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import mean_pinball_loss

from mqboost import MQDataset, MQRegressor


# 기본 설정
OUT_ROOT = "./0602_REALDATA_RESULT_randomsplit_100/mqboost_lgbm"
SHARED_ROOT = "./realdata_shared_randomsplit_repeat100"

DATASETS = ["forest_fires", "yacht_hydrodynamics", "airfoil"]
MODELS = ["mqboost_lgbm"]

TAU = np.arange(0.1, 1.0, 0.2)
N_REPEATS = 100
BASE_SEED = 100


# 공통 유틸: 첫 번째 real-data 코드 구조 유지
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"split_results": []}


def make_split_key(repeat_id):
    return f"repeat{repeat_id}"


def get_done_keys(split_results):
    return {make_split_key(fr["repeat"]) for fr in split_results}


def _safe_xy(X, y):
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
    return X, y


def mean_se(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, se


def per_tau_pinball(y_true, pred_mat, tau=TAU):
    y_true = np.asarray(y_true).reshape(-1)
    pred_mat = np.asarray(pred_mat, dtype=float)

    if pred_mat.ndim != 2:
        raise ValueError(f"pred_mat must be 2D, got shape={pred_mat.shape}")
    if pred_mat.shape[0] != len(y_true):
        raise ValueError(
            f"pred_mat row mismatch: pred_mat.shape={pred_mat.shape}, len(y_true)={len(y_true)}"
        )
    if pred_mat.shape[1] != len(tau):
        raise ValueError(
            f"pred_mat tau mismatch: pred_mat.shape={pred_mat.shape}, len(tau)={len(tau)}"
        )

    losses = []
    for j, q in enumerate(tau):
        losses.append(mean_pinball_loss(y_true, pred_mat[:, j], alpha=float(q)))
    return np.array(losses, dtype=float)


def composite_from_per_tau(per_tau_losses):
    return float(np.mean(per_tau_losses))


def crossing_percentage_from_preds(pred_mat):
    pred_mat = np.asarray(pred_mat)
    if pred_mat.ndim != 2 or pred_mat.shape[1] < 2:
        return 0.0
    crosses = (pred_mat[:, :-1] > pred_mat[:, 1:]).astype(np.float32)
    return float(crosses.mean() * 100.0)


def get_shared_paths(dataset_name, shared_root=SHARED_ROOT):
    dataset_dir = os.path.join(shared_root, dataset_name)
    return {
        "dir": dataset_dir,
        "data_csv": os.path.join(dataset_dir, f"{dataset_name}_full_data.csv"),
        "split_json": os.path.join(dataset_dir, f"{dataset_name}_splits_random.json"),
        "meta_json": os.path.join(dataset_dir, f"{dataset_name}_meta_random.json"),
    }


def load_shared_dataset_and_splits(dataset_name, shared_root=SHARED_ROOT):
    paths = get_shared_paths(dataset_name, shared_root)

    full_df = pd.read_csv(paths["data_csv"])

    with open(paths["split_json"], "r", encoding="utf-8") as f:
        split_info = json.load(f)

    with open(paths["meta_json"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    return full_df, split_info, meta


def subset_by_row_id(full_df, row_ids):
    df = full_df[full_df["row_id"].isin(row_ids)].copy()
    df = df.set_index("row_id").loc[row_ids].reset_index()

    y = df["y"].to_numpy(dtype=np.float64)
    X = df.drop(columns=["row_id", "y"]).to_numpy(dtype=np.float64)

    return _safe_xy(X, y)


# MQBoost-LGBM 파라미터: 두 번째 시나리오 MQBoost 코드 기준
def suggest_mqboost_lgbm_params(trial):
    return {
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.025, 0.05, 0.1, 0.2, 0.3]
        ),
        "num_leaves": trial.suggest_categorical(
            "num_leaves", [3, 7, 15, 31, 127]
        ),
        "top_rate": trial.suggest_categorical(
            "top_rate", [0.2, 0.4, 0.6, 0.7]
        ),
        "other_rate": trial.suggest_categorical(
            "other_rate", [0.05, 0.1, 0.3]
        ),
        "feature_fraction_bynode": trial.suggest_categorical(
            "feature_fraction_bynode", [0.25, 1.0, "sqrt", "log2"]
        ),
        "min_child_samples": trial.suggest_categorical(
            "min_child_samples", [1, 5, 25, 50, 70]
        ),
    }


def _normalize_feature_fraction_bynode(x, p):
    if x == "sqrt":
        return max(1.0 / p, math.sqrt(p) / p)
    if x == "log2":
        return max(1.0 / p, math.log2(p) / p)
    return float(x)


def _make_mq_lgbm_params(raw_params, p, seed, num_threads):
    params = raw_params.copy()
    params["feature_fraction_bynode"] = _normalize_feature_fraction_bynode(
        params["feature_fraction_bynode"], p
    )
    params["seed"] = int(seed)
    params["num_threads"] = int(num_threads)
    params["verbosity"] = -1
    params["boosting_type"] = "goss"
    params.pop("num_boost_round", None)
    return params


def fit_mqboost_lgbm(
    Xtr,
    ytr,
    Xpred,
    tau,
    params,
    seed,
    Xeval=None,
    yeval=None,
    num_boost_round=1000,
    num_threads=1,
):
    """
    MQBoost-LGBM 학습 후 Xpred에 대해 (n_samples, n_tau) 예측 행렬 반환.
    시나리오 코드의 MQDataset/MQRegressor 구조를 real data split에 그대로 적용.
    """
    tau = np.asarray(tau, dtype=float)
    Xtr, ytr = _safe_xy(Xtr, ytr)
    Xpred = np.ascontiguousarray(Xpred, dtype=np.float64)
    p = Xtr.shape[1]

    lgbm_params = _make_mq_lgbm_params(
        raw_params=params,
        p=p,
        seed=seed,
        num_threads=num_threads,
    )

    mq_regressor = MQRegressor(
        params=lgbm_params,
        objective="huber",
        model="lightgbm",
        epsilon=1e-4,
    )

    train_dataset = MQDataset(
        data=Xtr,
        label=ytr,
        alphas=list(tau),
        model="lightgbm",
    )

    if Xeval is not None and yeval is not None:
        Xeval, yeval = _safe_xy(Xeval, yeval)
        eval_dataset = MQDataset(
            data=Xeval,
            label=yeval,
            alphas=list(tau),
            model="lightgbm",
        )
        mq_regressor.fit(
            dataset=train_dataset,
            eval_set=eval_dataset,
            num_boost_round=int(num_boost_round),
        )
    else:
        mq_regressor.fit(
            dataset=train_dataset,
            num_boost_round=int(num_boost_round),
        )

    pred_dataset = MQDataset(
        data=Xpred,
        alphas=list(tau),
        model="lightgbm",
    )

    preds = np.asarray(mq_regressor.predict(pred_dataset), dtype=float)

    # MQBoost 버전에 따라 (K, n) 또는 (n, K)가 나올 수 있어서 보정
    if preds.ndim == 1:
        if len(tau) == 1:
            preds = preds.reshape(-1, 1)
        else:
            raise ValueError(f"1D prediction returned unexpectedly: shape={preds.shape}")

    if preds.shape[0] == len(tau) and preds.shape[1] == Xpred.shape[0]:
        preds = preds.T

    if preds.shape[0] != Xpred.shape[0] or preds.shape[1] != len(tau):
        raise ValueError(
            f"Prediction shape is wrong: preds.shape={preds.shape}, "
            f"expected=({Xpred.shape[0]}, {len(tau)})"
        )

    return preds


# MQBoost fit_predict: 첫 번째 real-data LGBM의 fit_predict_lgbm 역할
def fit_predict_mqboost_lgbm(
    Xtr,
    ytr,
    Xva,
    yva,
    Xfit,
    yfit,
    Xte,
    n_trials,
    seed,
    tau=TAU,
    num_boost_round=1000,
    num_threads=1,
    optuna_storage=None,
    study_name=None,
):
    tau = np.asarray(tau, dtype=float)

    def objective(trial):
        raw = suggest_mqboost_lgbm_params(trial)

        val_preds = fit_mqboost_lgbm(
            Xtr=Xtr,
            ytr=ytr,
            Xpred=Xva,
            tau=tau,
            params=raw,
            seed=seed,
            Xeval=Xva,
            yeval=yva,
            num_boost_round=num_boost_round,
            num_threads=num_threads,
        )

        return composite_from_per_tau(per_tau_pinball(yva, val_preds, tau))

    study_kwargs = {
        "direction": "minimize",
        "sampler": TPESampler(seed=int(seed)),
    }
    if optuna_storage is not None and study_name is not None:
        study_kwargs.update({
            "storage": optuna_storage,
            "study_name": study_name,
            "load_if_exists": True,
        })

    study = optuna.create_study(**study_kwargs)

    # storage를 쓰는 경우 이미 완료된 trial 수만큼 이어서 수행
    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    remaining_trials = max(0, int(n_trials) - len(complete_trials))

    print(
        f"  [Optuna] complete={len(complete_trials)}, "
        f"target={n_trials}, remaining={remaining_trials}"
    )

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)

    best = study.best_params.copy()
    best_valid = float(study.best_value)

    # 중요: 첫 번째 real-data 코드처럼 최종 test 예측은 Xfit/yfit 전체 train으로 재학습
    test_preds = fit_mqboost_lgbm(
        Xtr=Xfit,
        ytr=yfit,
        Xpred=Xte,
        tau=tau,
        params=best,
        seed=seed,
        Xeval=None,
        yeval=None,
        num_boost_round=num_boost_round,
        num_threads=num_threads,
    )

    return test_preds, best, best_valid


# 모델 라우터
def fit_predict_model(
    model_name,
    Xtr,
    ytr,
    Xva,
    yva,
    Xfit,
    yfit,
    Xte,
    n_trials,
    seed,
    tau=TAU,
    num_boost_round=1000,
    num_threads=1,
    optuna_storage=None,
    study_name=None,
):
    if model_name == "mqboost_lgbm":
        return fit_predict_mqboost_lgbm(
            Xtr=Xtr,
            ytr=ytr,
            Xva=Xva,
            yva=yva,
            Xfit=Xfit,
            yfit=yfit,
            Xte=Xte,
            n_trials=n_trials,
            seed=seed,
            tau=tau,
            num_boost_round=num_boost_round,
            num_threads=num_threads,
            optuna_storage=optuna_storage,
            study_name=study_name,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


# 메인 실행: 첫 번째 real-data 코드 구조 유지
def run_one(
    dataset_name,
    model_name,
    n_trials,
    num_boost_round,
    num_threads,
    shared_root,
    out_root,
    use_optuna_storage=True,
):
    print("\n" + "=" * 100)
    print(
        f"[START] dataset={dataset_name}, model={model_name}, "
        f"n_trials={n_trials}, num_boost_round={num_boost_round}"
    )
    print("=" * 100)

    full_df, split_info, meta = load_shared_dataset_and_splits(
        dataset_name,
        shared_root=shared_root,
    )

    out_dir = os.path.join(out_root, model_name, dataset_name)
    ensure_dir(out_dir)

    checkpoint_json = os.path.join(out_dir, f"{dataset_name}_{model_name}_result.json")
    result = load_checkpoint(checkpoint_json)

    result.update({
        "dataset_name": dataset_name,
        "model_name": model_name,
        "tau": [float(q) for q in TAU],
        "n_repeats": int(N_REPEATS),
        "n_trials": int(n_trials),
        "num_boost_round": int(num_boost_round),
        "mqboost_objective": "huber",
        "mqboost_model": "lightgbm",
        "meta": meta,
    })

    done_keys = get_done_keys(result["split_results"])

    if use_optuna_storage:
        optuna_storage = (
            f"sqlite:///{os.path.abspath(os.path.join(out_dir, f'optuna_{dataset_name}_{model_name}.db'))}"
        )
    else:
        optuna_storage = None

    for sp in split_info["splits"]:
        repeat_id = int(sp["repeat"])
        key = make_split_key(repeat_id)

        if key in done_keys:
            print(f"[SKIP] {dataset_name} | {model_name} | repeat={repeat_id}")
            continue

        print(f"[RUN] {dataset_name} | {model_name} | repeat={repeat_id}")

        repeat_seed = int(sp.get("repeat_seed", BASE_SEED + repeat_id * 1000))

        # 첫 번째 real-data 코드와 동일한 split 사용
        Xfit, yfit = subset_by_row_id(full_df, sp["train_row_id"])
        Xte, yte = subset_by_row_id(full_df, sp["test_row_id"])
        Xtr, ytr = subset_by_row_id(full_df, sp["inner_train_row_id"])
        Xva, yva = subset_by_row_id(full_df, sp["inner_valid_row_id"])

        study_name = f"{dataset_name}_{model_name}_repeat{repeat_id}"

        pred_mat, best_params, best_valid = fit_predict_model(
            model_name=model_name,
            Xtr=Xtr,
            ytr=ytr,
            Xva=Xva,
            yva=yva,
            Xfit=Xfit,
            yfit=yfit,
            Xte=Xte,
            n_trials=n_trials,
            seed=repeat_seed,
            tau=TAU,
            num_boost_round=num_boost_round,
            num_threads=num_threads,
            optuna_storage=optuna_storage,
            study_name=study_name,
        )

        per_tau_losses = per_tau_pinball(yte, pred_mat, TAU)
        composite = composite_from_per_tau(per_tau_losses)
        crossing = crossing_percentage_from_preds(pred_mat)

        split_result = {
            "repeat": repeat_id,
            "repeat_seed": repeat_seed,
            "best_valid_composite": float(best_valid),
            "best_params": best_params,
            "test_per_tau_pinball": {
                str(float(q)): float(v)
                for q, v in zip(TAU, per_tau_losses)
            },
            "test_composite_pinball": float(composite),
            "test_crossing_percentage": float(crossing),
            "n_train": int(len(yfit)),
            "n_test": int(len(yte)),
            "n_inner_train": int(len(ytr)),
            "n_inner_valid": int(len(yva)),
        }

        result["split_results"].append(split_result)

        composites = [r["test_composite_pinball"] for r in result["split_results"]]
        result["summary_so_far"] = {
            "completed_repeats": int(len(result["split_results"])),
            "composite_mean": mean_se(composites)[0],
            "composite_se": mean_se(composites)[1],
        }

        save_json(result, checkpoint_json)

        print(
            f"[DONE] repeat={repeat_id} | "
            f"valid_best={best_valid:.6f} | "
            f"test_comp={composite:.6f} | crossing={crossing:.3f}%"
        )

    # 최종 요약
    all_per_tau = []
    all_comp = []
    all_cross = []

    for r in result["split_results"]:
        all_comp.append(r["test_composite_pinball"])
        all_cross.append(r["test_crossing_percentage"])
        all_per_tau.append([
            r["test_per_tau_pinball"][str(float(q))]
            for q in TAU
        ])

    all_per_tau = np.asarray(all_per_tau, dtype=float)

    if len(all_comp) > 0:
        final_summary = {
            "completed_repeats": int(len(result["split_results"])),
            "composite_mean": mean_se(all_comp)[0],
            "composite_se": mean_se(all_comp)[1],
            "crossing_mean": mean_se(all_cross)[0],
            "crossing_se": mean_se(all_cross)[1],
            "per_tau_mean": {
                str(float(q)): float(all_per_tau[:, j].mean())
                for j, q in enumerate(TAU)
            },
            "per_tau_se": {
                str(float(q)): float(all_per_tau[:, j].std(ddof=1) / np.sqrt(all_per_tau.shape[0]))
                if all_per_tau.shape[0] > 1 else 0.0
                for j, q in enumerate(TAU)
            },
        }
    else:
        final_summary = {"completed_repeats": 0}

    result["final_summary"] = final_summary
    save_json(result, checkpoint_json)

    print(f"[FINISH] saved: {checkpoint_json}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num_boost_round",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--shared_root",
        type=str,
        default=SHARED_ROOT,
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default=OUT_ROOT,
    )

    parser.add_argument(
        "--no_optuna_storage",
        action="store_true",
        help="Use in-memory Optuna studies instead of sqlite resume storage.",
    )

    args = parser.parse_args()

    for dataset_name in args.datasets:
        for model_name in args.models:
            run_one(
                dataset_name=dataset_name,
                model_name=model_name,
                n_trials=args.n_trials,
                num_boost_round=args.num_boost_round,
                num_threads=args.num_threads,
                shared_root=args.shared_root,
                out_root=args.out_root,
                use_optuna_storage=not args.no_optuna_storage,
            )


if __name__ == "__main__":
    main()
