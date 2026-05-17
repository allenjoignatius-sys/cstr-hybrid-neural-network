"""
Script 3: Speed Benchmark & Final Evaluation
Compares NN inference speed vs ODE solver.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, json, time, joblib

from scipy.integrate import odeint
from sklearn.metrics import r2_score, mean_squared_error

# --- Redefine ODEs for standalone eval ---

def ode_irreversible(C, t, F, V, C_A0, k, n):
    C_A, C_B = C
    rate = k * max(C_A, 0) ** n
    return [(F/V)*(C_A0-C_A)-rate, (F/V)*(0.-C_B)+rate]

def ode_reversible(C, t, F, V, C_A0, kf, n_f, kr, n_r):
    C_A, C_B = C
    net = kf*max(C_A,0)**n_f - kr*max(C_B,0)**n_r
    return [(F/V)*(C_A0-C_A)-net, (F/V)*(0.-C_B)+net]

def ode_parallel(C, t, F, V, C_A0, k1, n1, k2, n2):
    C_A, C_B, C_C = C
    r1 = k1*max(C_A,0)**n1; r2 = k2*max(C_A,0)**n2
    return [(F/V)*(C_A0-C_A)-r1-r2, (F/V)*(0.-C_B)+r1, (F/V)*(0.-C_C)+r2]

# --- Helpers ---

def prompt_int(msg, default):
    try:
        raw = input(f"{msg} [default={default}]: ").strip()
        return int(raw) if raw else default
    except ValueError:
        return default

def load_artefacts():
    required = ["models/mlp_model.pkl", "models/scaler_X.pkl",
                "models/scaler_y.pkl", "models/model_meta.json",
                "data/run_params.json"]
    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}. Run scripts 1 and 2 first.")

    mlp = joblib.load("models/mlp_model.pkl")
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    with open("models/model_meta.json") as f: meta = json.load(f)
    with open("data/run_params.json") as f: run_cfg = json.load(f)
    return mlp, scaler_X, scaler_y, meta, run_cfg

def run_ode_batch(F_arr, CA0_arr, run_cfg):
    scheme = run_cfg["scheme"]
    V = run_cfg["V"]
    kinetics = run_cfg["kinetics"]
    time_span = np.linspace(0, run_cfg.get("t_end", 200.0), run_cfg.get("n_points", 200))

    results = []
    for F, CA0 in zip(F_arr, CA0_arr):
        if scheme == 1:
            sol = odeint(ode_irreversible, [0., 0.], time_span,
                         args=(F, V, CA0, kinetics["k"], kinetics["n"]))
            results.append(sol[-1])
        elif scheme == 2:
            sol = odeint(ode_reversible, [CA0, 0.], time_span,
                         args=(F, V, CA0, kinetics["kf"], kinetics["n_f"],
                               kinetics["kr"], kinetics["n_r"]))
            results.append(sol[-1])
        else:
            sol = odeint(ode_parallel, [CA0, 0., 0.], time_span,
                         args=(F, V, CA0, kinetics["k1"], kinetics["n1"],
                               kinetics["k2"], kinetics["n2"]))
            results.append(sol[-1])
    return np.array(results)

def benchmark(n_bench, mlp, scaler_X, scaler_y, meta, run_cfg):
    np.random.seed(99)
    F_arr = np.random.uniform(run_cfg["F_min"], run_cfg["F_max"], n_bench)
    CA0_arr = np.random.uniform(run_cfg["CA0_min"], run_cfg["CA0_max"], n_bench)
    feature_cols = meta["feature_cols"]

    X_bench = pd.DataFrame({
        "Flow_Rate_L_min": F_arr, "Inlet_Conc_mol_L": CA0_arr
    })[feature_cols].values

    # Timing ODE
    print(f"\nRunning {n_bench} ODE solves...")
    t0 = time.perf_counter()
    y_ode = run_ode_batch(F_arr, CA0_arr, run_cfg)
    ode_time = time.perf_counter() - t0

    # Timing NN
    print(f"Running {n_bench} NN inferences...")
    t0 = time.perf_counter()
    y_pred_sc = mlp.predict(scaler_X.transform(X_bench))
    if y_pred_sc.ndim == 1:
        y_pred_sc = y_pred_sc.reshape(-1, 1)
    y_nn = scaler_y.inverse_transform(y_pred_sc)
    nn_time = time.perf_counter() - t0

    speedup = ode_time / nn_time
    print(f"Speedup: {speedup:.1f}x faster")

    # Reconstruct targets for R2 score
    scheme = run_cfg["scheme"]
    records = []
    for i, (F, CA0) in enumerate(zip(F_arr, CA0_arr)):
        raw = y_ode[i]
        if scheme == 1 or scheme == 2:
            CA_ss, CB_ss = raw[0], raw[1]
            records.append([CA_ss, CB_ss, (CA0 - CA_ss) / CA0 if CA0 > 0 else 0])
        else:
            CA_ss, CB_ss, CC_ss = raw[0], raw[1], raw[2]
            records.append([CA_ss, CB_ss, CC_ss, CB_ss / (CB_ss + CC_ss + 1e-12)])

    ode_df = pd.DataFrame(records, columns=meta["target_cols"])
    y_true = ode_df[meta["target_cols"]].values

    r2 = r2_score(y_true, y_nn)
    rmse = np.sqrt(mean_squared_error(y_true, y_nn))
    print(f"Overall R2: {r2:.4f}")

    return F_arr, CA0_arr, y_true, y_nn, ode_time, nn_time, speedup, r2, rmse

def build_report_figure(y_true, y_nn, target_cols, ode_time, nn_time, speedup, r2, n_bench, out_dir):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Benchmark Report", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_speed = fig.add_subplot(gs[0, 0])
    bars = ax_speed.bar(["ODE", "NN"], [ode_time * 1000, nn_time * 1000], color=["#64748b", "#2563eb"])
    ax_speed.set_ylabel("Time (ms)")
    ax_speed.set_yscale("log")
    ax_speed.set_title(f"Speed ({n_bench} runs)")
    ax_speed.bar_label(bars, fmt="%.1f ms", padding=3)

    n_targets = y_true.shape[1]
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for i in range(min(n_targets, 5)):
        r, c = positions[i]
        ax = fig.add_subplot(gs[r, c])
        yt, yp = y_true[:, i], y_nn[:, i]
        ax.scatter(yt, yp, alpha=0.25, s=6)
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--")
        ax.set_title(f"{target_cols[i]}")
        ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "benchmark_report.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def main():
    print("\n--- Benchmark & Eval ---")
    mlp, scaler_X, scaler_y, meta, run_cfg = load_artefacts()
    n_bench = prompt_int("Num predictions to benchmark", 1000)

    _, _, y_true, y_nn, ode_time, nn_time, speedup, r2, _ = benchmark(
        n_bench, mlp, scaler_X, scaler_y, meta, run_cfg
    )

    os.makedirs("results", exist_ok=True)
    path = build_report_figure(y_true, y_nn, meta["target_cols"], ode_time, nn_time, speedup, r2, n_bench, "results")
    print(f"Report saved to {path}")

if __name__ == "__main__":
    main()