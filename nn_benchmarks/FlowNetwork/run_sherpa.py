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

path = pathlib.Path.cwd()



def run_sherpa(data_type ,rinv):
    parameters = [
        sherpa.Continuous("learning_rate", [1e-5, 1e-3], "log"),
        sherpa.Continuous("dropout", [0, 0.5]),
        sherpa.Ordinal("dense_layers", [1, 2, 3, 4, 5, 6, 7, 8]),
        sherpa.Ordinal("batch_size", [32, 64, 128]),
        sherpa.Discrete("dense_units", [20, 200]),
    ]
    
    results = path.parent / "results" / "sherpa" / "PFN" / data_type / rinv
    results.mkdir(parents=True, exist_ok=True)
    pathlib.Path('/tmp/sub1/sub2').mkdir(parents=True, exist_ok=True)

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(
        max_concurrent=1,
        model_type="GP_MCMC",
        acquisition_type="EI_MCMC",
        max_num_trials=max_num_trials,
    )

    study = sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=False,
        disable_dashboard=True,
        output_dir=results_path,
    )

    X, y = get_data(run_type, data_type, rinv, N)
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
            input_shape = None
        model = get_model(run_type, tp, input_shape=input_shape)
        for i in range(trial_epochs):
            model.fit(
                X_train, y_train, batch_size=int(tp["batch_size"]), verbose=0
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
    run_types = "HL"
    data_type = "CP"
    rinvs = ["0p0", "0p3", "1p0"]
    N = 50000
    max_num_trials = 50
    trial_epochs = 15
    for run_type in run_types:
        for rinv in rinvs:
            run_sherpa(run_type, data_type, rinv)

