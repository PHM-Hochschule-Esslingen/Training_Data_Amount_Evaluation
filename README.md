# Training Data Amount Evaluation

This repository contains the dataset and source code accompanying the paper *"Estimating Sufficient Amount of Training Data for Data-Driven Fault Diagnosis: Learning Curve Extrapolation and Entropy-Based Stopping Criteria"*.

The paper investigates two complementary approaches to determine whether a given dataset is sufficient for training data-driven fault diagnosis models, evaluated on a rolling bearing fault classification use case:

- **Approach I — Learning Curve Extrapolation:** Fits parametric functions (algebraic root, arctan, exponential, logarithmic, power law) to the initial segment of a learning curve and extrapolates them to estimate the amount of training data required to achieve a target classification accuracy.
- **Approach II — Entropy Analysis:** A training-free, online stopping criterion based on the Gradient of Normalized Entropy (GNE) that monitors the information gain from newly acquired data to decide whether further data acquisition is necessary.

## Contents

- **Dataset:** Rolling bearing vibration measurements from two test rigs covering healthy bearings as well as inner race, outer race, and rolling element faults under different speeds and radial loads (600 samples, 10-dimensional feature space, 4 classes).
- **Source code:** Scripts to reproduce the learning curves, fitting and extrapolation routines, and the entropy-based GNE evaluation presented in the paper.

# License
GNU General Public License v3.0

# Citation
If you use the dataset or the code provided in this repository, please cite:

Neu, M.; Braig, M.; Mauthe, F.; Zeiler, P. (2026): Estimating Sufficient Amount of Training Data for Data-Driven Fault Diagnosis: Learning Curve Extrapolation and Entropy-Based Stopping Criteria. International Journal of Prognostics and Health Management (submitted)
