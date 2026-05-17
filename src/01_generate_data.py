"""
Script 1: First-Principles Model & Dataset Generation
Supports irreversible (A->B), reversible (A<->B), and parallel (A->B, A->C) reactions.
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os
import time

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

def prompt_choice(msg, options):
    print(f"\n{msg}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        try:
            raw = input("Select number: ").strip()
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass

# --- ODE Definitions ---

def ode_irreversible(C, t, F, V, C_A0, k, n):
    C_A, C_B = C
    rate = k * max(C_A, 0) ** n
    dCa = (F / V) * (C_A0 - C_A) - rate
    dCb = (F / V) * (0.0  - C_B) + rate
    return [dCa, dCb]

def ode_reversible(C, t, F, V, C_A0, kf, n_f, kr, n_r):
    C_A, C_B = C
    net = kf * max(C_A, 0) ** n_f - kr * max(C_B, 0) ** n_r
    dCa = (F / V) * (C_A0 - C_A) - net
    dCb = (F / V) * (0.0  - C_B) + net
    return [dCa, dCb]

def ode_parallel(C, t, F, V, C_A0, k1, n1, k2, n2):
    C_A, C_B, C_C = C
    r1 = k1 * max(C_A, 0) ** n1
    r2 = k2 * max(C_A, 0) ** n2
    dCa = (F / V) * (C_A0 - C_A) - r1 - r2
    dCb = (F / V) * (0.0  - C_B) + r1
    dCc = (F / V) * (0.0  - C_C) + r2
    return [dCa, dCb, dCc]

# --- Config ---

def collect_parameters():
    print("\n--- CSTR Parameter Setup ---")
    print("Press Enter to use defaults.\n")

    scheme = prompt_choice(
        "Reaction scheme:",
        ["Irreversible (A->B)", "Reversible (A<->B)", "Parallel (A->B, A->C)"]
    )

    print("\n-- Geometry & Kinetics --")
    V = prompt_float("Reactor volume V (L)", 100.0)

    if scheme == 1:
        k  = prompt_float("Rate constant k", 0.10)
        n  = prompt_float("Reaction order n", 1.0)
        kinetics = dict(k=k, n=n)
    elif scheme == 2:
        kf  = prompt_float("Forward rate kf", 0.10)
        n_f = prompt_float("Forward order nf", 1.0)
        kr  = prompt_float("Reverse rate kr", 0.02)
        n_r = prompt_float("Reverse order nr", 1.0)
        kinetics = dict(kf=kf, n_f=n_f, kr=kr, n_r=n_r)
    else:
        k1  = prompt_float("Rate k1 (A->B)", 0.07)
        n1  = prompt_float("Order n1", 1.0)
        k2  = prompt_float("Rate k2 (A->C)", 0.03)
        n2  = prompt_float("Order n2", 2.0)
        kinetics = dict(k1=k1, n1=n1, k2=k2, n2=n2)

    print("\n-- Operating Range --")
    F_min   = prompt_float("Min flow rate (L/min)", 10.0)
    F_max   = prompt_float("Max flow rate (L/min)", 100.0)
    CA0_min = prompt_float("Min inlet conc (mol/L)", 0.5)
    CA0_max = prompt_float("Max inlet conc (mol/L)", 5.0)

    print("\n-- Dataset Config --")
    n_samples = prompt_int("Num samples", 5000)
    t_end     = prompt_float("Sim end time (min)", 200.0)
    n_points  = prompt_int("Time grid points", 200)
    seed      = prompt_int("Random seed", 42)

    return {
        "scheme": scheme, "V": V, "kinetics": kinetics,
        "F_min": F_min, "F_max": F_max, "CA0_min": CA0_min, "CA0_max": CA0_max,
        "n_samples": n_samples, "t_end": t_end, "n_points": n_points, "seed": seed,
    }

def run_simulation(params):
    scheme = params["scheme"]
    V = params["V"]
    kinetics = params["kinetics"]
    n_samples = params["n_samples"]
    time_span = np.linspace(0, params["t_end"], params["n_points"])

    np.random.seed(params["seed"])
    F_arr   = np.random.uniform(params["F_min"], params["F_max"], n_samples)
    CA0_arr = np.random.uniform(params["CA0_min"], params["CA0_max"], n_samples)

    records = []
    start_time = time.perf_counter()
    print(f"\nRunning {n_samples} ODE sims...")

    for i in range(n_samples):
        F = F_arr[i]
        CA0 = CA0_arr[i]

        if scheme == 1:
            sol = odeint(ode_irreversible, [0.0, 0.0], time_span,
                         args=(F, V, CA0, kinetics["k"], kinetics["n"]))
            CA_ss, CB_ss = sol[-1]
            row = {"Flow_Rate_L_min": F, "Inlet_Conc_mol_L": CA0,
                   "CA_ss_mol_L": CA_ss, "CB_ss_mol_L": CB_ss,
                   "Conversion": (CA0 - CA_ss) / CA0 if CA0 > 0 else 0}
        elif scheme == 2:
            sol = odeint(ode_reversible, [CA0, 0.0], time_span,
                         args=(F, V, CA0, kinetics["kf"], kinetics["n_f"],
                               kinetics["kr"], kinetics["n_r"]))
            CA_ss, CB_ss = sol[-1]
            row = {"Flow_Rate_L_min": F, "Inlet_Conc_mol_L": CA0,
                   "CA_ss_mol_L": CA_ss, "CB_ss_mol_L": CB_ss,
                   "Equilibrium_Conversion": (CA0 - CA_ss) / CA0 if CA0 > 0 else 0}
        else:
            sol = odeint(ode_parallel, [CA0, 0.0, 0.0], time_span,
                         args=(F, V, CA0, kinetics["k1"], kinetics["n1"],
                               kinetics["k2"], kinetics["n2"]))
            CA_ss, CB_ss, CC_ss = sol[-1]
            row = {"Flow_Rate_L_min": F, "Inlet_Conc_mol_L": CA0,
                   "CA_ss_mol_L": CA_ss, "CB_ss_mol_L": CB_ss,
                   "CC_ss_mol_L": CC_ss,
                   "Selectivity_B": CB_ss / (CB_ss + CC_ss + 1e-12)}
        records.append(row)

    elapsed = time.perf_counter() - start_time
    print(f"Done in {elapsed:.2f}s")
    return pd.DataFrame(records), elapsed

def main():
    params = collect_parameters()
    df, elapsed = run_simulation(params)

    os.makedirs("data", exist_ok=True)
    out_path = "data/reactor_data.csv"
    df.to_csv(out_path, index=False)

    import json
    log = {k: (v if not isinstance(v, dict) else v) for k, v in params.items()}
    with open("data/run_params.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Saved dataset to {out_path} ({len(df)} rows)")
    print(f"Saved config to data/run_params.json")

if __name__ == "__main__":
    main()