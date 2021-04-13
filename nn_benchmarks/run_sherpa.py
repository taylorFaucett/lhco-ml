# Standard Imports
import os
import sys
import pandas as pd
import h5py
import numpy as np
import sherpa
import tqdm
from sklearn.model_selection import train_test_split
import pathlib

# Import homemade tools
from get_model import get_model
from get_data import get_data

path = pathlib.Path.cwd()


def get_param(run_type):
    if run_type == "CNN_1" or "CNN_2":
        parameters = [
            sherpa.Continuous("learning_rate", [1e-5, 1e-3], "log"),
            sherpa.Continuous("dropout_1", [0, 0.5]),
            sherpa.Continuous("dropout_2", [0, 0.5]),
            sherpa.Discrete("filter_units_1", [32, 320]),
            sherpa.Discrete("filter_units_2", [32, 320]),
            sherpa.Discrete("filter_units_3", [32, 320]),
            sherpa.Discrete("filter_units_4", [32, 320]),
            sherpa.Discrete("max_pool_1", [2, 4]),
            sherpa.Discrete("max_pool_2", [2, 4]),
            sherpa.Discrete("kernel_size_1", [1, 4]),
            sherpa.Discrete("kernel_size_2", [1, 4]),
            sherpa.Discrete("kernel_size_3", [1, 4]),
            sherpa.Discrete("dense_units_1", [20, 300]),
            sherpa.Discrete("dense_units_2", [20, 300]),
        ]
    elif run_type == "HL":
        parameters = [
            sherpa.Continuous("learning_rate", [1e-5, 1e-3], "log"),
            sherpa.Continuous("dropout", [0, 0.5]),
            sherpa.Ordinal("dense_layers", [1, 2, 3, 4, 5, 6, 7, 8]),
            sherpa.Discrete("dense_units", [20, 200]),
        ]
    return parameters


def run_sherpa(run_type ,rinv, nb_chan):
    results_path = path / "results" / "sherpa" / run_type / rinv
    if not results_path.parent.exists():
        os.mkdir(results_path.parent)
    if not results_path.exists():
        os.mkdir(results_path)

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(
        max_concurrent=1,
        model_type="GP_MCMC",
        acquisition_type="EI_MCMC",
        max_num_trials=max_num_trials,
    )

    # algorithm = sherpa.algorithms.RandomSearch(max_num_trials=max_num_trials)
    parameters = get_param(run_type)

    study = sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=False,
        disable_dashboard=True,
        output_dir=results_path,
    )

    X, y = get_data(run_type, rinv, nb_chan, N)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.75, random_state=42
    )
    t = tqdm.tqdm(study, total=max_num_trials)
    for trial in t:
        # Sherpa settings in trials
        tp = trial.parameters
        if run_type == "HL":
            input_shape = X_train.shape[1]
        else:
            input_shape = (32, 32, X.shape[-1])
        model = get_model(run_type, tp, input_shape)
        for i in range(trial_epochs):
            model.fit(
                X_train, y_train, batch_size=128, verbose=0
            )
            loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
            study.add_observation(
                trial=trial,
                iteration=i,
                objective=auc,
                context={"loss": loss, "auc": auc, "accuracy": accuracy},
            )
            if study.should_trial_stop(trial):
                study.finalize(trial=trial, status="STOPPED")
                break

        study.finalize(trial=trial, status="COMPLETED")
        study.save()
        t.set_description(
            f"Trial {trial.id}; rinv={rinv.replace('p','.')} -> AUC = {auc:.4}"
        )


if __name__ == "__main__":
    run_type = sys.argv[1]
    if run_type == "CNN_1" or "CNN_2":
        nb_chan = int(run_type.split("_")[-1])
    else:
        nb_chan = 1
    rinvs = ["0p0", "0p3", "1p0"]
    N = 50000
    max_num_trials = 50
    trial_epochs = 5
    for rinv in rinvs:
        run_sherpa(run_type, rinv, nb_chan)
