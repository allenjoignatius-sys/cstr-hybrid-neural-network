# CSTR Hybrid Neural Network

> **A machine learning surrogate model for predicting Continuous Stirred-Tank Reactor (CSTR) steady-state concentrations, designed to bypass computationally expensive ODE solvers for real-time Model Predictive Control (MPC) applications.**

## Executive Summary
Traditional first-principles reactor models rely on complex Ordinary Differential Equations (ODEs) that are computationally expensive to solve in real-time. This project demonstrates a "Hybrid Modeling" approach by training a Multi-Layer Perceptron (MLP) Neural Network on data generated from classical thermodynamic simulations. 

The resulting AI surrogate model achieves near-perfect accuracy while offering orders of magnitude faster inference times, proving its viability for deployment in Advanced Process Control (APC) and digital twin environments.

## The Engineering Model
The foundational "ground truth" data is generated using a vectorised simulation of a non-isothermal CSTR. The general mass balance equation solved by `scipy.integrate.odeint` is:

$$\frac{dC_A}{dt} = \frac{F}{V}(C_{A0} - C_A) - r$$

Where:
* $F$ = Volumetric flow rate
* $V$ = Reactor volume
* $C_{A0}$ = Inlet concentration of species A
* $r$ = Reaction rate (supports irreversible $n$-th order, reversible, and parallel kinetics)

## Project Architecture
```text
cstr-hybrid-neural-network/
├── data/                   # Generated datasets and run configurations
├── models/                 # Serialised scikit-learn models and scalers (.pkl)
├── results/                # Visualisations (Parity plots, loss curves, benchmarks)
├── src/
│   ├── 01_generate_data.py # First-principles ODE simulation
│   ├── 02_train_model.py   # MLPRegressor training and hyperparameter tuning
│   └── 03_evaluate.py      # Inference benchmarking and parity plotting
├── requirements.txt        # Environment dependencies
└── README.md
