"""
Docstring for hyperparam
"""
import shutil
from pathlib import Path
import sys
import subprocess
import pandas as pd
import config as cfg
import numpy as np
from file_operations import create_directory, reset_directory
import optuna
from mnist.mnist import mnist_data

############################# NLSD ##################################
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
    file_path = Path("config.py")
    lines = file_path.read_text().splitlines()
    lines[4] = "epochs = 300"
    lines[5] = "save_every = 600"
    i = 0
    for cp0x in cfg.cp0_x:
        for cp1x in cfg.cp1_x:
            for cp1y in cfg.cp1_y:
                for length in cfg.l:
                    for w_0 in cfg.w0:
                        for w_1 in cfg.w1:
                            for w_2 in cfg.w2:
                                for w_3 in cfg.w3:
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
                                        # holdparam = f'{cp0}, {cp1}, {cp2}, {cfg.w0}, {cfg.w1}, {cfg.w2}, {cfg.w3}, {df['loss']}'
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
    cp0x = trial.suggest_float("cp0x", min(cfg.range_cp0_x), max(cfg.range_cp0_x))
    cp1x = trial.suggest_float("cp1x", min(cfg.range_cp1_x), max(cfg.range_cp1_x))
    cp1y = trial.suggest_float("cp1y", min(cfg.range_cp1_y), max(cfg.range_cp1_y))
    length = trial.suggest_float("length", min(cfg.range_l), max(cfg.range_l))

    w_0 = trial.suggest_float("w0", min(cfg.range_w0), max(cfg.range_w0))
    w_1 = trial.suggest_float("w1", min(cfg.range_w1), max(cfg.range_w1))
    w_2 = trial.suggest_float("w2", min(cfg.range_w2), max(cfg.range_w2))
    w_3 = trial.suggest_float("w3", min(cfg.range_w3), max(cfg.range_w3))

    lr = trial.suggest_float("learning_rate", min(cfg.range_lr), max(cfg.range_lr), log=True)

    # -------------------------
    # Derived parameters
    # -------------------------
    cp0 = [cp0x, 0.0]
    cp1 = [cp1x, cp1y]
    cp2 = [length / 1.4142, length / 1.4142]

    # -------------------------
    # Write config file
    # -------------------------
    directory = "./RESULTS/MNIST/"
    file_path = Path(f"{directory}nele={trial.number}/config.py")
    lines = file_path.read_text().splitlines()
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
    lines[48] = (f"AF_nele = ['nele={trial_id}']")
    file_path.write_text("\n".join(lines) + "\n")
    configfile = {}
    with open(f"{directory}nele={trial.number}/config.py") as f:
            exec(f.read(), configfile)
    best_val, trial =  mnist_data(f"{directory}nele={trial.number}", device, 2, configfile, f'nele={trial_id}', trial)
    return best_val

def run_study_mnist(device):
    directory = 'RESULTS/MNIST/'
    create_directory('RESULTS/MNIST/')

    source_file = "./config.py"

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        ),
    )
    for nt in range(cfg.TRIALS):
        directory = f"./RESULTS/MNIST/nele={nt}"
        create_directory(directory)
        shutil.copy(source_file, directory)
    
    study.optimize(
        lambda trial: objective(trial, device),
        n_trials=cfg.TRIALS,
        n_jobs=cfg.N_JOBS  # increase if you have GPUs/CPUs
    )

    print("Best validation loss:", study.best_value)
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {round(v, 4)}")

    return study