# Standard Imports
import os
import sys

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from rich import print
import tensorflow as tf
import pathlib
import gc

path = pathlib.Path.cwd()

# Import homemade tools
from get_data import get_data
from get_model import get_model
from mean_ci import mean_ci
from get_best_sherpa_result import get_best_sherpa_result

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def run_bootstraps(run_type, rinv, nb_chan, verbose, save_pred=False):
    # Trainig parameters from the sherpa optimization
    tp = get_best_sherpa_result(run_type, rinv)
    if verbose > 0:
        print("Training Parameters")
        print(tp)
    X, y = get_data(run_type, rinv, nb_chan, 290880)
    rs = ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.10)
    rs.get_n_splits(X)
    ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.10)
    straps = []
    aucs = []
    boot_ix = 0
    t = tqdm(list(rs.split(X)))

    for train_index, test_index in t:
        if run_type == "HL":
            input_shape = X.shape[1]
        else:
            input_shape = (32, 32, X.shape[-1])

        if run_type == "CNN":
            bs_path = path / "results" / "bootstrap" / f"{run_type}_{nb_chan}" / rinv
        else:
            bs_path = path / "results" / "bootstrap" / run_type / rinv
        model_file = bs_path / "models" / f"bs_{boot_ix}.h5"
        roc_file = bs_path / "roc" / f"roc_{boot_ix}.csv"
        ll_pred_file = bs_path / "ll_predictions" / f"ll_predictions_{boot_ix}.npy"

        if not bs_path.parent.exists():
            os.mkdir(bs_path.parent)
        if not bs_path.exists():
            os.mkdir(bs_path)
        if not model_file.parent.exists():
            os.mkdir(model_file.parent)
        if not roc_file.parent.exists():
            os.mkdir(roc_file.parent)
        if not ll_pred_file.parent.exists():
            os.mkdir(ll_pred_file.parent)

        if not model_file.exists():
            model = get_model(run_type, tp, input_shape)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    patience=5,
                    min_delta=0.0001,
                    verbose=verbose,
                    restore_best_weights=True,
                    mode="max",
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(model_file), monitor="val_auc", mode="max", verbose=verbose, save_best_only=True
                ),
            ]
            try:
                model.fit(
                    X[train_index],
                    y[train_index],
                    epochs=epochs,
                    verbose=verbose,
                    batch_size=128,
                    validation_data=(X[test_index], y[test_index]),
                    callbacks=callbacks,
                )
            except KeyboardInterrupt:
                print(f"removing file: {str(model_file)}")
                os.remove(str(model_file))
                sys.exit()
        else:
            model = tf.keras.models.load_model(str(model_file))


        val_predictions = np.hstack(model.predict(X[test_index]))
        auc_val = roc_auc_score(y[test_index], val_predictions)

        # Save the predictions
        if save_pred:
            np.save(ll_pred_file, model.predict(X))

        straps.append(boot_ix)
        aucs.append(auc_val)

        fpr, tpr, _ = roc_curve(y[test_index], val_predictions)
        roc_df = pd.DataFrame(
            {
                "fpr": fpr,
                "tpr": tpr,
            }
        )
        roc_df.to_csv(roc_file)
        boot_ix += 1


if __name__ == "__main__":
    run_type = str(sys.argv[1])
    rinv = str(sys.argv[2])
    if run_type == "CNN_1" or "CNN_2":
        nb_chan = int(run_type.split("_")[-1])
    else:
        nb_chan = 1
    n_splits = 200
    epochs = 200
    run_bootstraps(run_type, rinv, nb_chan=nb_chan, verbose=2, save_pred=True)
