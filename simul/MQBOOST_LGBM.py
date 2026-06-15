import os
import math
import numpy as np
import optuna
from optuna.samplers import TPESampler

from mqboost import MQDataset, MQRegressor

from new_exp_common import (
    composite_pinball,
    make_data,
    init_empty_result,
    load_res_json,
    add_summary_to_res,
    summarize_result,
    save_res_json,
)


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
            "min_child_samples", [1, 2, 5, 10, 15, 20, 25, 30]
        ),
    }


def _normalize_feature_fraction_bynode(x, p):
    if x == "sqrt":
        return max(1.0 / p, math.sqrt(p) / p)
    if x == "log2":
        return max(1.0 / p, math.log2(p) / p)
    return float(x)


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
):
    p = Xtr.shape[1]

    lgbm_params = params.copy()
    lgbm_params["feature_fraction_bynode"] = _normalize_feature_fraction_bynode(
        lgbm_params["feature_fraction_bynode"], p
    )

    lgbm_params["seed"] = seed
    lgbm_params["num_threads"] = 1
    lgbm_params["verbosity"] = -1
    lgbm_params["boosting_type"] = "goss"

    lgbm_params.pop("num_boost_round", None)

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
        eval_dataset = MQDataset(
            data=Xeval,
            label=yeval,
            alphas=list(tau),
            model="lightgbm",
        )

        mq_regressor.fit(
            dataset=train_dataset,
            eval_set=eval_dataset,
            num_boost_round=num_boost_round,
        )
    else:
        mq_regressor.fit(
            dataset=train_dataset,
            num_boost_round=num_boost_round,
        )

    pred_dataset = MQDataset(
        data=Xpred,
        alphas=list(tau),
        model="lightgbm",
    )

    preds = mq_regressor.predict(pred_dataset)
    preds = np.asarray(preds)

    if preds.shape[0] == len(tau):
        preds = preds.T

    if preds.shape[1] != len(tau):
        raise ValueError(
            f"Prediction shape is wrong: preds.shape={preds.shape}, len(tau)={len(tau)}"
        )

    return preds


def tune_mqboost_lgbm(
    Xtr,
    ytr,
    Xva,
    yva,
    tau,
    n_trials,
    seed,
    storage,
    study_name,
    num_boost_round=1000,
):
    def objective(trial):
        params = suggest_mqboost_lgbm_params(trial)

        val_preds = fit_mqboost_lgbm(
            Xtr=Xtr,
            ytr=ytr,
            Xpred=Xva,
            tau=tau,
            params=params,
            seed=seed,
            Xeval=Xva,
            yeval=yva,
            num_boost_round=num_boost_round,
        )

        val_comp, _ = composite_pinball(yva, val_preds, tau)
        return val_comp

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    n_done = len(complete_trials)
    remaining_trials = max(0, n_trials - n_done)

    print(
        f"  [Optuna] study={study_name}, "
        f"complete={n_done}, target={n_trials}, remaining={remaining_trials}"
    )

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            show_progress_bar=False,
        )

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print(
        f"  [Optuna done] complete={len(complete_trials)}, "
        f"best_value={study.best_value:.6f}"
    )

    return study.best_params.copy(), study.best_value


def run_one_setting(
    scenario_id=1,
    p=1,
    train_n=1000,
    mc_repeats=100,
    tau=np.arange(0.1, 1.0, 0.2),
    n_trials=100,
    num_boost_round=1000,
    out_dir="./optuna_results_mqboost_lgbm",
):
    model_tag = "mqboost_lgbm"

    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(
        out_dir,
        f"res_{model_tag}_s{scenario_id}_p{p}.json"
    )

    storage = (
        f"sqlite:///{os.path.abspath(os.path.join(out_dir, f'optuna_{model_tag}_s{scenario_id}_p{p}.db'))}"
    )

    if os.path.exists(save_path):
        print(f"[Resume] loading existing result: {save_path}")
        out = load_res_json(save_path)

        out["model"]["comp"] = list(out["model"]["comp"])
        out["model"]["per_tau"] = [
            np.array(x, dtype=float)
            for x in out["model"]["per_tau"]
        ]

        if "best_params" not in out:
            out["best_params"] = []
        if "best_val" not in out:
            out["best_val"] = []

    else:
        out = init_empty_result(tau)
        out["best_params"] = []
        out["best_val"] = []

    start_rep = len(out["model"]["comp"])
    print(f"[Progress] completed repetitions: {start_rep}/{mc_repeats}")

    K = len(tau)

    for i in range(start_rep, mc_repeats):
        print(f"[scenario={scenario_id} | p={p}] repetition {i}/{mc_repeats - 1}")

        train, valid, test = make_data(
            scenario_id,
            train_n=train_n,
            seed=i,
            p=p,
        )

        Xtr = train["data"]
        ytr = np.asarray(train["label"]).reshape(-1)

        Xva = valid["data"]
        yva = np.asarray(valid["label"]).reshape(-1)

        Xte = test["data"]
        yte = np.asarray(test["label"]).reshape(-1)

        study_name = f"{model_tag}_s{scenario_id}_p{p}_rep{i}"

        best_params, best_val = tune_mqboost_lgbm(
            Xtr=Xtr,
            ytr=ytr,
            Xva=Xva,
            yva=yva,
            tau=tau,
            n_trials=n_trials,
            seed=i,
            storage=storage,
            study_name=study_name,
            num_boost_round=num_boost_round,
        )

        test_preds = fit_mqboost_lgbm(
            Xtr=Xtr,
            ytr=ytr,
            Xpred=Xte,
            tau=tau,
            params=best_params,
            seed=i,
            Xeval=Xva,
            yeval=yva,
            num_boost_round=num_boost_round,
        )

        test_comp, test_tau = composite_pinball(
            yte,
            test_preds,
            tau,
        )

        out["model"]["comp"].append(float(test_comp))
        out["model"]["per_tau"].append(np.array(test_tau, dtype=float))
        out["best_params"].append(best_params)
        out["best_val"].append(float(best_val))

        print(
            f"  MQBoost-LGBM: val(best)={best_val:.6f} "
            f"test(comp)={test_comp:.6f} best={best_params}"
        )

        temp_out = {
            "model": {
                "comp": np.array(out["model"]["comp"], dtype=float),
                "per_tau": (
                    np.vstack(out["model"]["per_tau"])
                    if len(out["model"]["per_tau"]) > 0
                    else np.empty((0, K))
                ),
            },
            "best_params": out["best_params"],
            "best_val": out["best_val"],
            "tau": np.array(tau, dtype=float),
            "n_trials": n_trials,
            "num_boost_round": num_boost_round,
        }

        temp_out = add_summary_to_res(temp_out)

        save_res_json(
            temp_out,
            out_dir,
            scenario_id,
            p,
            model_tag,
        )

        print(f"  [Checkpoint saved] repetition {i} complete")

    out["model"]["comp"] = np.array(out["model"]["comp"], dtype=float)
    out["model"]["per_tau"] = (
        np.vstack(out["model"]["per_tau"])
        if len(out["model"]["per_tau"]) > 0
        else np.empty((0, K))
    )

    out["tau"] = np.array(tau, dtype=float)
    out["n_trials"] = n_trials
    out["num_boost_round"] = num_boost_round

    out = add_summary_to_res(out)

    save_res_json(
        out,
        out_dir,
        scenario_id,
        p,
        model_tag,
    )

    return out


if __name__ == "__main__":
    tau = np.arange(0.1, 1.0, 0.2)

    for scenario_id in [1]:
        res = run_one_setting(
            scenario_id=scenario_id,
            p=1,
            train_n=1000,
            mc_repeats=100,
            tau=tau,
            n_trials=100,
            num_boost_round=1000,
            out_dir=f"./0608mq_huber/optuna_results_mqboost_lgbm_eval_s{scenario_id}_p1",
        )

        summarize_result(
            res,
            model_name=f"MQBoost-LGBM_s{scenario_id}_p1",
        )