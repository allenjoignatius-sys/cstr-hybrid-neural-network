"""
=============================================================================
CSTR Hybrid Neural Network Project
Script 1: First-Principles Model & Dataset Generation
=============================================================================
Supports generalised nth-order reactions.  All operating parameters are
supplied interactively at runtime so no hard-coded numbers need editing.
 
Supported reaction schemes
---------------------------
  1. Irreversible  A → B          rate = k · Cₐⁿ
  2. Reversible    A ⇌ B          rate = kf · Cₐⁿ − kr · Cbᵐ
  3. Parallel      A → B, A → C   rate_B = k1 · Cₐⁿ¹,  rate_C = k2 · Cₐⁿ²
=============================================================================
"""
 
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os
import time
 
# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def prompt_float(msg, default):
    """Ask the user for a float; use *default* if they just press Enter."""
    try:
        raw = input(f"  {msg} [default={default}]: ").strip()
        return float(raw) if raw else default
    except ValueError:
        print(f"    ⚠  Invalid input — using default ({default}).")
        return default
 
def prompt_int(msg, default):
    """Ask the user for an int; use *default* if they just press Enter."""
    try:
        raw = input(f"  {msg} [default={default}]: ").strip()
        return int(raw) if raw else default
    except ValueError:
        print(f"    ⚠  Invalid input — using default ({default}).")
        return default
 
def prompt_choice(msg, options):
    """Ask the user to pick a numbered option from a list."""
    print(f"\n  {msg}")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        try:
            raw = input("  Enter number: ").strip()
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        print("    ⚠  Please enter a valid number.")
 
# ─────────────────────────────────────────────────────────────────────────────
# ODE DEFINITIONS  (one function per reaction scheme)
# ─────────────────────────────────────────────────────────────────────────────
 
def ode_irreversible(C, t, F, V, C_A0, k, n):
    """
    Irreversible A → B  (nth order in A)
    dCa/dt = (F/V)(Ca0 − Ca) − k·Ca^n
    dCb/dt = (F/V)(0  − Cb) + k·Ca^n        [Cb0 = 0 assumed]
    """
    C_A, C_B = C
    rate = k * max(C_A, 0) ** n
    dCa = (F / V) * (C_A0 - C_A) - rate
    dCb = (F / V) * (0.0  - C_B) + rate
    return [dCa, dCb]
 
def ode_reversible(C, t, F, V, C_A0, kf, n_f, kr, n_r):
    """
    Reversible A ⇌ B
    net rate = kf·Ca^nf − kr·Cb^nr
    """
    C_A, C_B = C
    net = kf * max(C_A, 0) ** n_f - kr * max(C_B, 0) ** n_r
    dCa = (F / V) * (C_A0 - C_A) - net
    dCb = (F / V) * (0.0  - C_B) + net
    return [dCa, dCb]
 
def ode_parallel(C, t, F, V, C_A0, k1, n1, k2, n2):
    """
    Parallel  A → B  (rate k1·Ca^n1)
              A → C  (rate k2·Ca^n2)
    """
    C_A, C_B, C_C = C
    r1 = k1 * max(C_A, 0) ** n1
    r2 = k2 * max(C_A, 0) ** n2
    dCa = (F / V) * (C_A0 - C_A) - r1 - r2
    dCb = (F / V) * (0.0  - C_B) + r1
    dCc = (F / V) * (0.0  - C_C) + r2
    return [dCa, dCb, dCc]
 
# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE PARAMETER COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
 
def collect_parameters():
    """Walk the user through every tuneable parameter."""
    print("\n" + "═"*60)
    print("  CSTR HYBRID MODEL — PARAMETER SETUP")
    print("═"*60)
    print("  Press Enter on any prompt to accept the default value.\n")
 
    # ── Reaction scheme ──────────────────────────────────────────────────────
    scheme = prompt_choice(
        "Choose a reaction scheme:",
        [
            "Irreversible  A → B   (nth order)",
            "Reversible    A ⇌ B   (forward nth / reverse mth order)",
            "Parallel      A → B and A → C",
        ]
    )
 
    # ── Reactor geometry ─────────────────────────────────────────────────────
    print("\n  ── Reactor Geometry ──────────────────────────")
    V = prompt_float("Reactor volume  V  (L)", 100.0)
 
    # ── Kinetic constants (depends on scheme) ────────────────────────────────
    print("\n  ── Kinetics ──────────────────────────────────")
    if scheme == 1:
        k  = prompt_float("Rate constant  k  (units depend on order)", 0.10)
        n  = prompt_float("Reaction order  n", 1.0)
        kinetics = dict(k=k, n=n)
 
    elif scheme == 2:
        kf  = prompt_float("Forward rate constant  kf", 0.10)
        n_f = prompt_float("Forward reaction order  nf", 1.0)
        kr  = prompt_float("Reverse rate constant  kr", 0.02)
        n_r = prompt_float("Reverse reaction order  nr", 1.0)
        kinetics = dict(kf=kf, n_f=n_f, kr=kr, n_r=n_r)
 
    else:  # scheme == 3
        k1  = prompt_float("Rate constant  k1 (A→B)", 0.07)
        n1  = prompt_float("Reaction order  n1", 1.0)
        k2  = prompt_float("Rate constant  k2 (A→C)", 0.03)
        n2  = prompt_float("Reaction order  n2", 2.0)
        kinetics = dict(k1=k1, n1=n1, k2=k2, n2=n2)
 
    # ── Operating ranges (training envelope) ─────────────────────────────────
    print("\n  ── Operating Envelope (training range) ───────")
    F_min   = prompt_float("Minimum flow rate  F_min  (L/min)", 10.0)
    F_max   = prompt_float("Maximum flow rate  F_max  (L/min)", 100.0)
    CA0_min = prompt_float("Minimum inlet concentration  Ca0_min  (mol/L)", 0.5)
    CA0_max = prompt_float("Maximum inlet concentration  Ca0_max  (mol/L)", 5.0)
 
    # ── Dataset size ─────────────────────────────────────────────────────────
    print("\n  ── Dataset ───────────────────────────────────")
    n_samples = prompt_int("Number of simulation samples", 5000)
    t_end     = prompt_float("Simulation end time  t_end  (min, for steady state)", 200.0)
    n_points  = prompt_int("Number of time-grid points", 200)
    seed      = prompt_int("Random seed (for reproducibility)", 42)
 
    return {
        "scheme": scheme,
        "V": V,
        "kinetics": kinetics,
        "F_min": F_min, "F_max": F_max,
        "CA0_min": CA0_min, "CA0_max": CA0_max,
        "n_samples": n_samples,
        "t_end": t_end,
        "n_points": n_points,
        "seed": seed,
    }
 
# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
 
def run_simulation(params):
    """Vectorised simulation over random operating conditions."""
    scheme   = params["scheme"]
    V        = params["V"]
    kinetics = params["kinetics"]
    n_samples = params["n_samples"]
    time_span = np.linspace(0, params["t_end"], params["n_points"])
 
    np.random.seed(params["seed"])
    F_arr   = np.random.uniform(params["F_min"],   params["F_max"],   n_samples)
    CA0_arr = np.random.uniform(params["CA0_min"], params["CA0_max"], n_samples)
 
    records = []
    start_time = time.perf_counter()
 
    print(f"\n  Running {n_samples:,} ODE simulations … ", end="", flush=True)
    for i in range(n_samples):
        F   = F_arr[i]
        CA0 = CA0_arr[i]
 
        if scheme == 1:
            y0  = [0.0, 0.0]
            sol = odeint(ode_irreversible, y0, time_span,
                         args=(F, V, CA0, kinetics["k"], kinetics["n"]))
            CA_ss, CB_ss = sol[-1]
            row = {"Flow_Rate_L_min": F, "Inlet_Conc_mol_L": CA0,
                   "CA_ss_mol_L": CA_ss, "CB_ss_mol_L": CB_ss,
                   "Conversion": (CA0 - CA_ss) / CA0 if CA0 > 0 else 0}
 
        elif scheme == 2:
            y0  = [CA0, 0.0]
            sol = odeint(ode_reversible, y0, time_span,
                         args=(F, V, CA0,
                               kinetics["kf"], kinetics["n_f"],
                               kinetics["kr"], kinetics["n_r"]))
            CA_ss, CB_ss = sol[-1]
            row = {"Flow_Rate_L_min": F, "Inlet_Conc_mol_L": CA0,
                   "CA_ss_mol_L": CA_ss, "CB_ss_mol_L": CB_ss,
                   "Equilibrium_Conversion": (CA0 - CA_ss) / CA0 if CA0 > 0 else 0}
 
        else:  # parallel
            y0  = [CA0, 0.0, 0.0]
            sol = odeint(ode_parallel, y0, time_span,
                         args=(F, V, CA0,
                               kinetics["k1"], kinetics["n1"],
                               kinetics["k2"], kinetics["n2"]))
            CA_ss, CB_ss, CC_ss = sol[-1]
            row = {"Flow_Rate_L_min": F, "Inlet_Conc_mol_L": CA0,
                   "CA_ss_mol_L": CA_ss, "CB_ss_mol_L": CB_ss,
                   "CC_ss_mol_L": CC_ss,
                   "Selectivity_B": CB_ss / (CB_ss + CC_ss + 1e-12)}
 
        records.append(row)
 
    elapsed = time.perf_counter() - start_time
    print(f"done in {elapsed:.1f}s")
 
    df = pd.DataFrame(records)
    return df, elapsed
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    params = collect_parameters()
    df, elapsed = run_simulation(params)
 
    os.makedirs("data", exist_ok=True)
    out_path = "data/reactor_data.csv"
    df.to_csv(out_path, index=False)
 
    # Save run parameters as a log for reproducibility
    import json
    log = {k: (v if not isinstance(v, dict) else v) for k, v in params.items()}
    with open("data/run_params.json", "w") as f:
        json.dump(log, f, indent=2)
 
    print(f"\n  ✅  Dataset saved  →  {out_path}")
    print(f"  ✅  Parameters log →  data/run_params.json")
    print(f"  Rows : {len(df):,}")
    print(f"  Cols : {list(df.columns)}")
    print(f"  ODE wall time : {elapsed:.2f}s  ({elapsed/len(df)*1000:.2f} ms/sample)\n")
    print(df.describe().round(4))
    print("\n  → Run  python src/02_train_model.py  to train the neural network.\n")
 
if __name__ == "__main__":
    main()
