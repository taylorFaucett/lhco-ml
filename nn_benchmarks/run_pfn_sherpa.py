# Standard Imports
import os
import sys
import pandas as pd
import h5py
import numpy as np
import sherpa
import tqdm
import tools
import tensorflow as tf
from sklearn.model_selection import train_test_split
from energyflow.archs import EFN, PFN
from energyflow.utils import data_split, to_categorical
from imblearn.over_sampling import SMOTE
import pathlib

# Import homemade tools
from get_model import get_model
from get_data import get_data

path = pathlib.Path.cwd()


def run_sherpa(network ,bbx):
    results_path = path / "results" / "sherpa" / network / bbx
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

    parameters = [
        sherpa.Continuous("learning_rate", [1e-5, 1e-3], "log"),
        sherpa.Continuous("latent_dropout", [0, 0.5]),
        sherpa.Continuous("f_dropout", [0, 0.5]),
        sherpa.Ordinal("batch_size", [32, 64, 128]),
        sherpa.Discrete("phi_size", [20, 200]),
        sherpa.Discrete("f_size", [20, 200]),
    ]


    study = sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=False,
        disable_dashboard=True,
        output_dir=results_path,
    )
    h5_file = h5py.File(path.parent / "data" / "with_rotations" / f"{bbx}.h5", "r")
    X = h5_file["features"][:]
    y = h5_file["targets"][:] 
    for x in tqdm.tqdm(X):
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
    
    if network == "PFN":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.75, random_state=42
        )
        input_dim = X_train.shape[-1]
    elif network == "EFN":
        z = X[:, :, 0]
        p = X[:, :, 1:]
        X = [z, p]
        
        z_train, z_test, y_train, y_test = train_test_split(
            z, y, test_size=0.75, random_state=42
        )
        p_train, p_test, y_train, y_test = train_test_split(
            p, y, test_size=0.75, random_state=42
        )
        X_train = [z_train, p_train]
        X_test = [z_test, p_test]
        input_dim = 3
    num_classes = len(list(set(y)))
    Y_train = to_categorical(y_train, num_classes=num_classes)
    Y_test = to_categorical(y_test, num_classes=num_classes)
    t = tqdm.tqdm(study, total=max_num_trials)
    for trial in t:
        # Sherpa settings in trials
        tp = trial.parameters
        model = eval(network)(
                input_dim=input_dim,
                Phi_sizes= (tp["phi_size"], tp["phi_size"], tp["phi_size"]), 
                F_sizes= (tp["f_size"], tp["f_size"], tp["f_size"]),
                Phi_acts="relu", 
                F_acts = "relu",
                Phi_k_inits="glorot_normal",
                F_k_inits="glorot_normal",
                latent_dropout=tp["latent_dropout"],
                F_dropouts=tp["f_dropout"],
                mask_val = 0,
                loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(lr=tp["learning_rate"]),
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
                output_act="softmax",
                summary=False
            )  
    
        for i in range(trial_epochs):
            model.fit(
                X_train, Y_train, batch_size=int(tp["batch_size"]), verbose=0
            )
            loss, accuracy, auc = model.evaluate(X_test, Y_test, verbose=0)
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
            f"Trial {trial.id}; blackbox={bbx} -> AUC = {auc:.4}"
        )


if __name__ == "__main__":
    N = 100000
    max_num_trials = 50
    trial_epochs = 15
    run_sherpa(str(sys.argv[1]), str(sys.argv[2]))
