# 2D Physics-Informed Neural Network for Glomerular Filtration

This repository provides a PyTorch implementation of a 2D Physics-Informed Neural Network (PINN) used to model fluid dynamics and solute transport within a simplified glomerular capillary. 

The model couples the steady-state Poiseuille flow equations with a transient Convection-Diffusion-Reaction (CDR) PDE to simulate the clearance of metabolic solutes (e.g., creatinine) across a semi-permeable membrane, governed by varying hydrostatic pressures and filtration coefficients.

## Mathematical Formulation

The domain is defined as a 2D capillary cross-section where $x \in [0, 1]$ represents the axial length and $y \in [0, 1]$ represents the radial width. The system solves the following PDE for solute concentration $C(x,y,t)$:

$$ \frac{\partial C}{\partial t} + u(y) \frac{\partial C}{\partial x} = D \left( \frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2} \right) - k C $$

Where:
* $u(y) = u_{max} \cdot (1 - (\frac{y - 0.5}{0.5})^2)$ defines the parabolic velocity profile.
* $D$ is the diffusion coefficient.
* $k$ is the bulk filtration rate.

### Boundary and Initial Conditions
* **IC:** $C(x,y,0) = 0$ (Initial state is solute-free).
* **Inlet:** $C(0,y,t) = 1$ (Constant solute influx).
* **Walls:** $C(x,0,t) = 0$ and $C(x,1,t) = 0$ (Dirichlet boundary conditions simulating instant clearance at the permeable basement membrane).

## Architecture & Optimization
The forward model is a continuous, fully-connected Multi-Layer Perceptron (MLP) parameterized by $\theta$.
* **Layers:** 3 inputs $(x,y,t)$, 4 hidden layers (40 units each), 1 output $(C)$.
* **Activation:** $\tanh$ (required for $C^2$ continuity to compute second-order spatial derivatives).
* **Loss formulation:** 
  $\mathcal{L}_{total} = \lambda \mathcal{L}_{data} + \mathcal{L}_{PDE}$
  where $\lambda = 200$ acts as a penalty coefficient to strictly enforce the boundary and initial conditions against the unsupervised PDE residual.

## Experiments

The `run_simulation.py` script executes two comparative states:

1. **Normotensive State (Healthy):** Uses $u_{max} = 1.0$ and $k = 1.5$. The concentration gradient demonstrates effective solute clearance along the capillary length.
2. **Hypertensive/Hyperfiltration State:** Uses $u_{max} = 3.5$ and $k = 0.1$. The velocity field overwhelms the filtration capacity, resulting in significant solute retention at the outlet.

## Requirements and Usage

Dependencies:
* `torch`
* `numpy`
* `matplotlib`

To train the models and generate the cross-sectional concentration plots at $t=1.0$:
```bash
pip install -r requirements.txt
python run_simulation.py
```
