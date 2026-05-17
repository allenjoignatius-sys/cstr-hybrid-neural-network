"""
=============================================================================
CSTR Hybrid Neural Network Project
Script 2: Neural Network Training
=============================================================================
Reads reactor_data.csv produced by 01_generate_data.py and trains an
MLPRegressor to predict steady-state concentrations directly from operating
conditions — bypassing the expensive ODE solver at inference time.
 
The user is prompted to choose which columns to use as features / targets,
network architecture, and training hyper-parameters.
=============================================================================
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, time, joblib
 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
 
# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def prompt_float(msg, default):
    try:
        raw = input(f"  {msg} [default={default}]: ").strip()
        return float(raw) if raw else default
    except ValueError:
        print(f"    ⚠  Invalid — using {default}")
        return default
 
def prompt_int(msg, default):
    try:
        raw = input(f"  {msg} [default={default}]: ").strip()
        return int(raw) if raw else default
    except ValueError:
        print(f"    ⚠  Invalid — using {default}")
        return default
 
def prompt_str(msg, default):
    raw = input(f"  {msg} [default={default}]: ").strip()
    return raw if raw else default
 
def prompt_choice(msg, options):
    print(f"\n  {msg}")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        try:
            c = int(input("  Enter number: ").strip())
            if 1 <= c <= len(options):
                return c
        except ValueError:
            pass
        print("    ⚠  Enter a valid number.")
 
def pick_columns(available, role="target", default=None):
    """Let the user pick which columns from the CSV to use."""
    if not available:
        return []
    if default is None:
        default = list(available)
    print(f"\n  Available columns for {role}:")
    for i, col in enumerate(available, 1):
        print(f"    [{i}] {col}")
    default_str = ",".join(str(available.index(c)+1) for c in default if c in available)
    raw = input(f"  Select column numbers (comma-separated) [default={default_str}]: ").strip()
    if not raw:
        return list(default)
    indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
    chosen = [available[i] for i in indices if 0 <= i < len(available)]
    return chosen if chosen else list(default)
 
# ─────────────────────────────────────────────────────────────────────────────
# NETWORK ARCHITECTURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
 
def build_hidden_layers():
    """
    Let the user define network depth and width interactively.
    Returns a tuple like (128, 64, 32).
    """
    print("\n  ── Network Architecture ──────────────────────")
    n_layers = prompt_int("Number of hidden layers", 3)
    layers = []
    for i in range(1, n_layers + 1):
        default_size = max(16, 128 // (2 ** (i - 1)))   # 128, 64, 32 …
        size = prompt_int(f"  Neurons in hidden layer {i}", default_size)
        layers.append(size)
    return tuple(layers)
 
# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
 
def train(df, feature_cols, target_cols, hidden_layers, hp):
    X = df[feature_cols].values
    y = df[target_cols].values
    if y.ndim == 1:
        y = y.reshape(-1, 1)
 
    # Scale both inputs and outputs for stable training
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled,
        test_size=hp["test_size"],
        random_state=hp["seed"]
    )
 
    print(f"\n  Training samples : {len(X_train):,}")
    print(f"  Test    samples  : {len(X_test):,}")
    print(f"  Architecture     : {hidden_layers}")
    print(f"  Max iterations   : {hp['max_iter']}")
    print(f"\n  Training … ", end="", flush=True)
 
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
    print(f"done in {elapsed:.1f}s  ({mlp.n_iter_} iterations)")
 
    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred_scaled = mlp.predict(X_test)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
 
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
 
    r2  = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
 
    print(f"\n  ── Test-Set Performance ──────────────────────")
    print(f"  R²   : {r2:.6f}")
    print(f"  RMSE : {rmse:.6f}  mol/L")
 
    return mlp, scaler_X, scaler_y, X_test, y_true, y_pred, r2, rmse, elapsed
 
# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_results(y_true, y_pred, target_cols, r2, out_dir):
    n_targets = y_true.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]
 
    for ax, col, yt, yp in zip(axes, target_cols, y_true.T, y_pred.T):
        ax.scatter(yt, yp, alpha=0.3, s=10, color="#2563eb", edgecolors="none")
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel(f"ODE Solver — {col}", fontsize=11)
        ax.set_ylabel(f"Neural Network — {col}", fontsize=11)
        ax.set_title(f"Parity Plot: {col}\nR² = {r2:.4f}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    path = os.path.join(out_dir, "parity_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊  Parity plot saved → {path}")
 
def plot_loss_curve(mlp, out_dir):
    plt.figure(figsize=(7, 4))
    plt.plot(mlp.loss_curve_, color="#2563eb", label="Training loss")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        # Convert validation scores (R²) to a comparable scale
        plt.plot(
            [1 - s for s in mlp.validation_scores_],
            color="#dc2626", linestyle="--", label="Validation loss proxy"
        )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊  Loss curve saved  → {path}")
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    print("\n" + "═"*60)
    print("  CSTR HYBRID MODEL — NEURAL NETWORK TRAINING")
    print("═"*60)
 
    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = prompt_str("Path to reactor_data.csv", "data/reactor_data.csv")
    if not os.path.exists(data_path):
        print(f"\n  ❌  File not found: {data_path}")
        print("     Run  python src/01_generate_data.py  first.\n")
        return
 
    df = pd.read_csv(data_path)
    print(f"\n  Loaded {len(df):,} rows  |  Columns: {list(df.columns)}")
 
# ── Feature / target selection ────────────────────────────────────────────
    all_cols = list(df.columns)

    # Sensible defaults: first 2 cols are always inputs, rest are outputs
    default_features = all_cols[:2]
    default_targets  = all_cols[2:]

    print("\n  ── Feature Selection ─────────────────────────")
    print(f"  (Suggested inputs: {default_features})")
    feature_cols = pick_columns(all_cols, "features (inputs to the NN)",
                                default=default_features)

    remaining = [c for c in all_cols if c not in feature_cols]
    print("\n  ── Target Selection ──────────────────────────")
    print(f"  (Suggested targets: {remaining})")
    target_cols = pick_columns(remaining, "targets (outputs to predict)",
                               default=remaining)

    # Guard: catch empty targets before anything breaks
    if not target_cols:
        print("\n  ❌  No target columns selected. "
              "You must leave at least one column out of features for the NN to predict.")
        return
     
    print(f"\n  Features : {feature_cols}")
    print(f"  Targets  : {target_cols}")
 
    # ── Architecture & hyper-parameters ──────────────────────────────────────
    hidden_layers = build_hidden_layers()
 
    print("\n  ── Hyper-parameters ──────────────────────────")
    activation_choice = prompt_choice(
        "Activation function:",
        ["relu (recommended)", "tanh", "logistic"]
    )
    activation = ["relu", "tanh", "logistic"][activation_choice - 1]
 
    solver_choice = prompt_choice(
        "Optimiser:",
        ["adam (recommended)", "lbfgs (good for small datasets)", "sgd"]
    )
    solver = ["adam", "lbfgs", "sgd"][solver_choice - 1]
 
    hp = {
        "activation": activation,
        "solver":     solver,
        "lr":         prompt_float("Learning rate (adam/sgd only)", 0.001),
        "max_iter":   prompt_int("Max iterations", 500),
        "test_size":  prompt_float("Test-set fraction", 0.20),
        "seed":       prompt_int("Random seed", 42),
    }
 
    # ── Train ─────────────────────────────────────────────────────────────────
    mlp, scaler_X, scaler_y, X_test, y_true, y_pred, r2, rmse, elapsed = \
        train(df, feature_cols, target_cols, hidden_layers, hp)
 
    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(mlp,      "models/mlp_model.pkl")
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")
 
    meta = {
        "feature_cols":   feature_cols,
        "target_cols":    target_cols,
        "hidden_layers":  list(hidden_layers),
        "hyper_params":   hp,
        "r2":             r2,
        "rmse":           rmse,
        "train_time_s":   elapsed,
        "n_iter":         mlp.n_iter_,
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
 
    print(f"\n  ✅  Model saved       → models/mlp_model.pkl")
    print(f"  ✅  Scalers saved     → models/scaler_X.pkl / scaler_y.pkl")
    print(f"  ✅  Metadata saved    → models/model_meta.json")
 
    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    plot_results(y_true, y_pred, target_cols, r2, "results")
    plot_loss_curve(mlp, "results")
 
    print(f"\n  → Run  python src/03_evaluate.py  for the speed benchmark.\n")
 
if __name__ == "__main__":
    main()
 
