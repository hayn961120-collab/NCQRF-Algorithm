import os
import numpy as np
import optuna
from optuna.samplers import TPESampler

from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

from new_exp_common import (
    composite_pinball,
    make_data,
    init_empty_result,
    load_res_json,
    add_summary_to_res,
    summarize_result,
    save_res_json,
)

def make_max_features_grid(p):
    if p == 1:
        return [1]

    vals = sorted(set([
        int(np.floor(np.sqrt(p))),
        int(np.floor(np.log2(p))),
        int(np.floor(0.25 * p)),
        p
    ]))
    vals = [v for v in vals if v >= 1]
    return vals

def suggest_ngboost_params(trial, p):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [0.025, 0.05, 0.1, 0.2, 0.3]),
        "max_depth": trial.suggest_categorical("max_depth", [2, 3, 5]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 5, 25, 50, 70]),
        "max_features": trial.suggest_categorical("max_features", make_max_features_grid(p)),
        "subsample": trial.suggest_categorical("subsample", [0.5, 0.75, 1.0]),
        "natural_gradient": trial.suggest_categorical("natural_gradient", [True]),
    }
def fit_ngboost_multi_quantile(Xtr, ytr, Xva, yva, Xte, tau, params, seed):
    base = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=seed,
    )

    model = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=base,
        natural_gradient=params["natural_gradient"],
        n_estimators=1000,
        learning_rate=params["learning_rate"],
        minibatch_frac=params["subsample"],
        random_state=seed,
        verbose=False,
    )

    model.fit(Xtr, ytr)

    val_dist = model.pred_dist(Xva)
    test_dist = model.pred_dist(Xte)

    val_preds = np.column_stack([val_dist.ppf(float(q)) for q in tau])
    test_preds = np.column_stack([test_dist.ppf(float(q)) for q in tau])

    val_comp, _ = composite_pinball(yva, val_preds, tau)
    return val_comp, test_preds


def tune_ngboost(Xtr, ytr, Xva, yva, tau, n_trials, seed, storage, study_name, p):
    def objective(trial):
        params = suggest_ngboost_params(trial, p)
        val_comp, _ = fit_ngboost_multi_quantile(
            Xtr, ytr, Xva, yva, Xva, tau, params, seed
        )
        return val_comp

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = max(0, n_trials - n_done)

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)

    return study.best_params.copy(), study.best_value


def run_one_setting(
    scenario_id=1,
    p=20,
    train_n=1000,
    mc_repeats=100,
    tau=np.arange(0.1, 1.0, 0.2),
    n_trials=100,
    out_dir="./optuna_results_ngboost",
):
    model_tag = "ngboost"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"res_{model_tag}_s{scenario_id}_p{p}.json")
    storage = f"sqlite:///{os.path.abspath(os.path.join(out_dir, f'optuna_{model_tag}_s{scenario_id}_p{p}.db'))}"

    if os.path.exists(save_path):
        print(f"[Resume] loading existing result: {save_path}")
        out = load_res_json(save_path)
        out["model"]["comp"] = list(out["model"]["comp"])
        out["model"]["per_tau"] = [np.array(x, dtype=float) for x in out["model"]["per_tau"]]
    else:
        out = init_empty_result(tau)

    start_rep = len(out["model"]["comp"])
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

        study_name = f"{model_tag}_s{scenario_id}_p{p}_rep{i}"
        best_params, best_val = tune_ngboost(
            Xtr, ytr, Xva, yva, tau, n_trials=n_trials, seed=i,
            storage=storage, study_name=study_name, p=p
        )

        _, test_preds = fit_ngboost_multi_quantile(Xtr, ytr, Xte, yte, Xte, tau, best_params, i)
        test_comp, test_tau = composite_pinball(yte, test_preds, tau)

        out["model"]["comp"].append(float(test_comp))
        out["model"]["per_tau"].append(np.array(test_tau, dtype=float))
        out["best_params"].append(best_params)
        out["best_val"].append(float(best_val))

        print(f"  NGBoost: val(best)={best_val:.6f} test(comp)={test_comp:.6f} best={best_params}")

        temp_out = {
            "model": {
                "comp": np.array(out["model"]["comp"], dtype=float),
                "per_tau": np.vstack(out["model"]["per_tau"]) if len(out["model"]["per_tau"]) > 0 else np.empty((0, K)),
            },
            "best_params": out["best_params"],
            "best_val": out["best_val"],
            "tau": np.array(tau, dtype=float),
        }

        temp_out = add_summary_to_res(temp_out)
        save_res_json(temp_out, out_dir, scenario_id, p, model_tag)
        print(f"  [Checkpoint saved] repetition {i} complete")

    out["model"]["comp"] = np.array(out["model"]["comp"], dtype=float)
    out["model"]["per_tau"] = np.vstack(out["model"]["per_tau"]) if len(out["model"]["per_tau"]) > 0 else np.empty((0, K))
    out["tau"] = np.array(tau, dtype=float)

    out = add_summary_to_res(out)
    save_res_json(out, out_dir, scenario_id, p, model_tag)
    return out


if __name__ == "__main__":
    tau = np.arange(0.1, 1.0, 0.2)

    res = run_one_setting(
        scenario_id=1,
        p=1,
        train_n=1000,
        mc_repeats=100,
        tau=tau,
        n_trials=100,
        out_dir="./0510_NGB/optuna_results_ngboost_s1_p1",
    )

    summarize_result(res, model_name="NGBoost")