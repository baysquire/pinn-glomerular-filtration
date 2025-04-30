# 2D Physics-Informed Neural Network for Glomerular Filtration

This repository provides a PyTorch implementation of a 2D Physics-Informed Neural Network (PINN) used to model fluid dynamics and solute transport within a simplified glomerular capillary. 

The model couples the steady-state Poiseuille flow equations with a transient Convection-Diffusion-Reaction (CDR) PDE to simulate the clearance of metabolic solutes (e.g., creatinine) across a semi-permeable membrane, governed by varying hydrostatic pressures and filtration coefficients.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   PINN Architecture                     │
│                                                         │
│  Inputs: (x, y, t)                                      │
│      │                                                   │
│      ▼                                                   │
│  ┌────────────────────────────────────────┐              │
│  │  Fully-Connected MLP                   │              │
│  │  [3] → [40] → [40] → [40] → [40] → [1]│              │
│  │  Activation: tanh (C² continuity)      │              │
│  └────────────────┬───────────────────────┘              │
│                   │                                      │
│                   ▼                                      │
│              Output: C(x,y,t)                            │
│                   │                                      │
│          ┌────────┴────────┐                             │
│          ▼                 ▼                              │
│   ┌─────────────┐  ┌──────────────────────┐             │
│   │  Data Loss   │  │  Physics (PDE) Loss   │             │
│   │  (BC + IC)   │  │  (Navier-Stokes +     │             │
│   │              │  │   Conv-Diff-Rxn)      │             │
│   └──────┬──────┘  └─────────┬────────────┘             │
│          │                    │                           │
│          ▼                    ▼                           │
│   Total Loss = 200·L_data + L_physics                    │
│          │                                               │
│          ▼                                               │
│     Adam Optimizer (backpropagation via autograd)         │
└─────────────────────────────────────────────────────────┘
```

## Mathematical Formulation

The domain is defined as a 2D capillary cross-section where $x \in [0, 1]$ represents the axial length and $y \in [0, 1]$ represents the radial width. The system solves the following PDE for solute concentration $C(x,y,t)$:

$$ \frac{\partial C}{\partial t} + u(y) \frac{\partial C}{\partial x} = D \left( \frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2} \right) - k C $$

Where:
* $u(y) = u_{max} \cdot (1 - (\frac{y - 0.5}{0.5})^2)$ defines the parabolic velocity profile (Poiseuille Flow).
* $D$ is the diffusion coefficient.
* $k$ is the bulk filtration rate.

### Boundary and Initial Conditions
* **IC:** $C(x,y,0) = 0$ (Initial state is solute-free).
* **Inlet:** $C(0,y,t) = 1$ (Constant solute influx).
* **Walls:** $C(x,0,t) = 0$ and $C(x,1,t) = 0$ (Dirichlet boundary conditions simulating instant clearance at the permeable basement membrane).

## Optimization

The forward model is a continuous, fully-connected Multi-Layer Perceptron (MLP) parameterized by $\theta$.
* **Layers:** 3 inputs $(x,y,t)$, 4 hidden layers (40 units each), 1 output $(C)$.
* **Activation:** $\tanh$ (required for $C^2$ continuity to compute second-order spatial derivatives).
* **Loss formulation:** 
  $\mathcal{L}_{total} = \lambda \mathcal{L}_{data} + \mathcal{L}_{PDE}$
  where $\lambda = 200$ acts as a penalty coefficient to strictly enforce the boundary and initial conditions against the unsupervised PDE residual.

## Experiments

The `run_simulation.py` script executes two comparative states:

| Experiment | $u_{max}$ | $k$ | Description |
|---|---|---|---|
| **Normotensive (Healthy)** | 1.0 | 1.5 | Effective solute clearance along the capillary length |
| **Hypertensive (Diseased)** | 3.5 | 0.1 | Velocity overwhelms filtration → solute retention at outlet |

### Training Configuration
* **Collocation points:** 4,000 (interior domain)
* **Boundary points:** 800 per condition (IC, inlet, wall×2)
* **Epochs:** 6,000 per experiment
* **Learning rate:** $1 \times 10^{-3}$
* **Optimizer:** Adam

## Results

### Normotensive State
The concentration gradient demonstrates effective solute clearance — toxins entering at the inlet ($x=0$) are progressively filtered and removed at the capillary walls.

### Hypertensive State
Under elevated hydrostatic pressure and impaired filtration ($k=0.1$), the axial convection dominates over transverse diffusion, resulting in significant solute retention at the outlet ($x=1$). This models the clinical phenomenon of hyperfiltration injury.

## Requirements and Usage

Dependencies:
* `torch>=2.0.0`
* `numpy>=1.24.0`
* `matplotlib>=3.7.0`
* `scipy>=1.10.0`

To train the models and generate the cross-sectional concentration plots at $t=1.0$:
```bash
pip install -r requirements.txt
python run_simulation.py
```

An interactive walkthrough is also available:
```bash
jupyter notebook Walkthrough.ipynb
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this work, please cite:
```bibtex
@software{sojobi2026pinn,
  author = {Sojobi, Abiodun},
  title = {2D Physics-Informed Neural Network for Glomerular Filtration},
  year = {2026},
  url = {https://github.com/baysquire/pinn-glomerular-filtration}
}
```
