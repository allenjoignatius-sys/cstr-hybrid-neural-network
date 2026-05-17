# CSTR Hybrid Neural Network

> **A machine learning surrogate model for predicting Continuous Stirred-Tank Reactor (CSTR) steady-state concentrations, designed to bypass computationally expensive ODE solvers for real-time Model Predictive Control (MPC) applications.**

## Executive Summary
Traditional first-principles reactor models rely on complex Ordinary Differential Equations (ODEs) that are computationally expensive to solve in real-time. This project demonstrates a "Hybrid Modeling" approach by training a Multi-Layer Perceptron (MLP) Neural Network on data generated from classical thermodynamic simulations. 

The resulting AI surrogate model achieves near-perfect accuracy while offering orders of magnitude faster inference times, proving its viability for deployment in Advanced Process Control (APC) and digital twin environments.

## The Engineering Model
The foundational "ground truth" data is generated using a vectorised simulation of a CSTR. The system solves the mass balance ODEs dynamically and supports three distinct, user-selectable kinetic mechanisms:

1. **Irreversible ($A \rightarrow B$):** General $n$-th order kinetics.
   $$r = k C_A^n$$
2. **Reversible ($A \rightleftharpoons B$):** Forward and reverse reactions with independent reaction orders.
   $$r_{net} = k_f C_A^{n_f} - k_r C_B^{n_r}$$
3. **Parallel ($A \rightarrow B, A \rightarrow C$):** Competing reactions allowing for selectivity analysis.
   $$r_1 = k_1 C_A^{n_1}, \quad r_2 = k_2 C_A^{n_2}$$

The general mass balance for the primary reactant ($A$) across all schemes is defined as:
$$\frac{dC_A}{dt} = \frac{F}{V}(C_{A0} - C_A) - \sum r_{A, consumed}$$

Where $F$ is volumetric flow rate, $V$ is reactor volume, and $C_{A0}$ is the inlet concentration.

## Project Architecture
```text
cstr-hybrid-neural-network/
├── data/                   # Generated datasets and run configurations
├── models/                 # Serialised scikit-learn models and scalers (.pkl)
├── results/                # Visualisations (Parity plots, loss curves, benchmarks)
├── src/
│   ├── 01_generate_data.py # Interactive first-principles ODE simulation
│   ├── 02_train_model.py   # MLPRegressor training and hyperparameter tuning
│   └── 03_evaluate.py      # Inference benchmarking and parity plotting
├── requirements.txt        # Environment dependencies
└── README.md
