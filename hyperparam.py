"""
Docstring for hyperparam
"""
from pathlib import Path
import sys
import subprocess
import pandas as pd
import config as cfg
import numpy as np
from file_operations import create_directory, reset_directory
import optuna
import time
from mnist.mnist import mnist_data

create_directory("./HYPERPARAM")
file_path = Path("config.py")
lines = file_path.read_text().splitlines()
cp0_x = [-1.0, -4.0, -6.0]
cp1_x = [-0.1, -0.3]
cp1_y = [-0.1, -0.3]
l = [-1.0, -0.5]
w0 = [1.0, 0.5, 1.9807]
w1 = [1.0, 1.5, 0.7178]
w2 = [1.0, 1.5, 0.1238]
w3 = [1.0, 1.5, 0.1981]

############################# NLSD ##################################
lines[4] = "epochs = 300"
lines[5] = "save_every = 600"
curve_loss = np.ones(len(cfg.FUNC_NLSD_NELE))
mincp0 = np.ones((len(cfg.FUNC_NLSD_NELE), 2))
mincp1 = np.ones((len(cfg.FUNC_NLSD_NELE), 2))
mincp2 = np.ones((len(cfg.FUNC_NLSD_NELE), 2))
minw0 = np.ones(len(cfg.FUNC_NLSD_NELE))
minw1 = np.ones(len(cfg.FUNC_NLSD_NELE))
minw2 = np.ones(len(cfg.FUNC_NLSD_NELE))
minw3 = np.ones(len(cfg.FUNC_NLSD_NELE))

def run_study(device):
    """Run hyper param tuning study
    """
    reset_directory("./HYPERPARAM/NLSD")
    i = 0
    for cp0x in cp0_x:
        for cp1x in cp1_x:
            for cp1y in cp1_y:
                for length in l:
                    for w_0 in w0:
                        for w_1 in w1:
                            for w_2 in w2:
                                for w_3 in w3:
                                    lines[6] = f"learning_rate = 0.01"
                                    cp0 = [cp0x, 0.0]
                                    cp1 = [cp1x, cp1y]
                                    cp2 = [length/1.4142, length/1.41422]
                                    print(f'Study:{i}')
                                    lines[9] = f"w0 = {w_0}"
                                    lines[10] = f"w1 = {w_1}"
                                    lines[11] = f"w2 = {w_2}"
                                    lines[12] = f"w3 = {w_3}"
                                    lines[13] = f"cp0 = [{cp0x}, {0.0}]"
                                    lines[14] = f"cp1 = [{cp1x}, {cp1y}]"
                                    lines[15] = (
                                        f"cp2 = [{length/1.4142},{length/1.4142}]")
                                    lines[33] = (
                                        f"AF_NLSD_NELE = ['nele={i}']"
                                    )
                                    file_path.write_text("\n".join(lines) + "\n")
                                    # launch the study
                                    
                                    subprocess.run(
                                        [sys.executable, "main.py", "--mode=train_nele", 
                                        f"--device={device}", "--type=nlsd"],
                                        check=True
                                    )

                                    for j, curve in enumerate(cfg.FUNC_NLSD_NELE):
                                        df = pd.read_csv(f"./RESULTS/NLSD/{curve}/nele={i}/loss_history_nele={i}.csv")
                                        # holdparam = f'{cp0}, {cp1}, {cp2}, {w0}, {w1}, {w2}, {w3}, {df['loss']}'
                                        row = pd.DataFrame([{
                                            "cp0": cp0,
                                            "cp1": cp1,
                                            "cp2": cp2,
                                            "w0": w_0,
                                            "w1": w_1,
                                            "w2": w_2,
                                            "w3": w_3,
                                            "loss": df["loss"].iloc[-1],
                                        }])
                                        row.to_csv(f"./HYPERPARAM/NLSD/hyperparam_{curve}.csv", mode="a", 
                                                header=not Path(f"./HYPERPARAM/NLSD/hyperparam_{curve}.csv").exists(), index=False)
                                    
                                    i += 1
                                    # sys.exit()
def find_min():
    """Find minimum from a list of csv
    """
    for j, curve in enumerate(cfg.FUNC_NLSD_NELE):                   
        df = pd.read_csv(f"./HYPERPARAM/NLSD/hyperparam_{curve}.csv")
        idx = df["loss"].idxmin()      # index of minimum loss
        best_row = df.loc[idx]         # full row
        print(f"{curve}")
        print(best_row)

def objective(trial, device):
    # -------------------------
    # Hyperparameters to tune
    # -------------------------
    cp0x = trial.suggest_float("cp0x", min(cp0_x), max(cp0_x))
    cp1x = trial.suggest_float("cp1x", min(cp1_x), max(cp1_x))
    cp1y = trial.suggest_float("cp1y", min(cp1_y), max(cp1_y))
    length = trial.suggest_float("length", min(l), max(l))

    w_0 = trial.suggest_float("w0", min(w0), max(w0))
    w_1 = trial.suggest_float("w1", min(w1), max(w1))
    w_2 = trial.suggest_float("w2", min(w2), max(w2))
    w_3 = trial.suggest_float("w3", min(w3), max(w3))

    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # -------------------------
    # Derived parameters
    # -------------------------
    cp0 = [cp0x, 0.0]
    cp1 = [cp1x, cp1y]
    cp2 = [length / 1.4142, length / 1.4142]

    # -------------------------
    # Write config file
    # -------------------------
    # reset_directory("./HYPERPARAM/MNIST")
    lines[4]  = f"epochs = 1"
    lines[5]  = f"save_every = 10"
    lines[6]  = f"learning_rate = {lr}"
    lines[7]  = f"val_ratio = 0.1"
    lines[9]  = f"w0 = {w_0}"
    lines[10] = f"w1 = {w_1}"
    lines[11] = f"w2 = {w_2}"
    lines[12] = f"w3 = {w_3}"
    lines[13] = f"cp0 = {cp0}"
    lines[14] = f"cp1 = {cp1}"
    lines[15] = f"cp2 = {cp2}"

    # -------------------------
    # Training loop (30 epochs)
    # -------------------------

    best_val = float('inf')
    trial_id = trial.number

    lines[48] = (
                f"AF_nele = ['nele={trial_id}']"
            )
    file_path.write_text("\n".join(lines) + "\n")
    create_directory("./RESULTS/MNIST/"+cfg.AF_nele[0])
    best_val, trial =  mnist_data(cfg.epochs, cfg.learning_rate, device, 2, cfg.AF_nele[0], trial)
    return best_val

def run_study_mnist(device):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        ),
    )

    study.optimize(
        lambda trial: objective(trial, device),
        n_trials=5,
        n_jobs=1  # increase if you have GPUs/CPUs
    )

    print("Best validation loss:", study.best_value)
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {round(v, 4)}")

    return study