"""
Script 2: Neural Network Training
Trains an MLPRegressor on the data generated in script 1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, time, joblib

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- Helpers ---

def prompt_float(msg, default):
    try:
        raw = input(f"{msg} [default={default}]: ").strip()
        return float(raw) if raw else default
    except ValueError:
        return default

def prompt_int(msg, default):
    try:
        raw = input(f"{msg} [default={default}]: ").strip()
        return int(raw) if raw else default
    except ValueError:
        return default

def prompt_str(msg, default):
    raw = input(f"{msg} [default={default}]: ").strip()
    return raw if raw else default

def prompt_choice(msg, options):
    print(f"\n{msg}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        try:
            c = int(input("Select number: ").strip())
            if 1 <= c <= len(options):
                return c
        except ValueError:
            pass

def pick_columns(available, role="target", default=None):
    if not available:
        return []
    if default is None:
        default = list(available)
    print(f"\nAvailable for {role}:")
    for i, col in enumerate(available, 1):
        print(f"  [{i}] {col}")
    default_str = ",".join(str(available.index(c)+1) for c in default if c in available)
    raw = input(f"Select cols (comma-separated) [default={default_str}]: ").strip()
    if not raw:
        return list(default)
    indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
    chosen = [available[i] for i in indices if 0 <= i < len(available)]
    return chosen if chosen else list(default)

def build_hidden_layers():
    print("\n-- Architecture --")
    n_layers = prompt_int("Num hidden layers", 3)
    layers = []
    for i in range(1, n_layers + 1):
        default_size = max(16, 128 // (2 ** (i - 1)))
        size = prompt_int(f"Neurons in layer {i}", default_size)
        layers.append(size)
    return tuple(layers)

# --- Training Logic ---

def train(df, feature_cols, target_cols, hidden_layers, hp):
    X = df[feature_cols].values
    y = df[target_cols].values
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=hp["test_size"], random_state=hp["seed"]
    )

    print(f"\nTraining on {len(X_train)} samples...")
    t0 = time.perf_counter()
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=hp["activation"],
        solver=hp["solver"],
        learning_rate_init=hp["lr"],
        max_iter=hp["max_iter"],
        random_state=hp["seed"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )
    mlp.fit(X_train, y_train if y_train.shape[1] > 1 else y_train.ravel())
    elapsed = time.perf_counter() - t0
    print(f"Finished in {elapsed:.1f}s ({mlp.n_iter_} iterations)")

    y_pred_scaled = mlp.predict(X_test)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    r2  = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n-- Test Results --")
    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f} mol/L")

    return mlp, scaler_X, scaler_y, X_test, y_true, y_pred, r2, rmse, elapsed

# --- Plotting ---

def plot_results(y_true, y_pred, target_cols, r2, out_dir):
    n_targets = y_true.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]

    for ax, col, yt, yp in zip(axes, target_cols, y_true.T, y_pred.T):
        ax.scatter(yt, yp, alpha=0.3, s=10, color="#2563eb", edgecolors="none")
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
        ax.set_xlabel(f"ODE - {col}")
        ax.set_ylabel(f"NN - {col}")
        ax.set_title(f"{col} (R2={r2:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "parity_plot.png"), dpi=150)
    plt.close()

def plot_loss_curve(mlp, out_dir):
    plt.figure(figsize=(7, 4))
    plt.plot(mlp.loss_curve_, color="#2563eb", label="Train loss")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        plt.plot(
            [1 - s for s in mlp.validation_scores_],
            color="#dc2626", linestyle="--", label="Val loss proxy"
        )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

def main():
    print("\n--- NN Training ---")
    data_path = prompt_str("Path to data", "data/reactor_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    all_cols = list(df.columns)
    default_features = all_cols[:2]
    default_targets  = all_cols[2:]

    feature_cols = pick_columns(all_cols, "features", default=default_features)
    remaining = [c for c in all_cols if c not in feature_cols]
    target_cols = pick_columns(remaining, "targets", default=remaining)

    if not target_cols:
        print("Error: No targets selected.")
        return

    hidden_layers = build_hidden_layers()

    print("\n-- Hyperparams --")
    activation = ["relu", "tanh", "logistic"][prompt_choice("Activation:", ["relu", "tanh", "logistic"]) - 1]
    solver = ["adam", "lbfgs", "sgd"][prompt_choice("Solver:", ["adam", "lbfgs", "sgd"]) - 1]

    hp = {
        "activation": activation,
        "solver": solver,
        "lr": prompt_float("Learning rate", 0.001),
        "max_iter": prompt_int("Max iterations", 500),
        "test_size": prompt_float("Test split", 0.20),
        "seed": prompt_int("Random seed", 42),
    }

    mlp, scaler_X, scaler_y, X_test, y_true, y_pred, r2, rmse, elapsed = train(
        df, feature_cols, target_cols, hidden_layers, hp
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(mlp, "models/mlp_model.pkl")
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")

    meta = {
        "feature_cols": feature_cols, "target_cols": target_cols,
        "hidden_layers": list(hidden_layers), "hyper_params": hp,
        "r2": r2, "rmse": rmse, "train_time_s": elapsed, "n_iter": mlp.n_iter_,
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    os.makedirs("results", exist_ok=True)
    plot_results(y_true, y_pred, target_cols, r2, "results")
    plot_loss_curve(mlp, "results")
    print("Saved models and plots.")

if __name__ == "__main__":
    main()