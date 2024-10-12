# 2D Physics-Informed Neural Network for Glomerular Filtration

A PyTorch implementation of a 2D PINN for modeling fluid dynamics and solute
transport within a simplified glomerular capillary.

The model couples the steady-state Poiseuille flow equations with a transient
Convection-Diffusion-Reaction (CDR) PDE.

## Mathematical Formulation

The solute concentration C(x,y,t) satisfies:

dC/dt + u(y) * dC/dx = D * (d2C/dx2 + d2C/dy2) - k * C

Where u(y) is the parabolic velocity profile (Poiseuille Flow).

## Usage

```bash
pip install -r requirements.txt
python run_simulation.py
```

An interactive walkthrough is also available:
```bash
jupyter notebook Walkthrough.ipynb
```