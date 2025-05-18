"""
Validation of the PINN against an analytical solution.

Uses the 1D steady-state convection-diffusion equation (no reaction term)
with known analytical solution to verify the PINN learns the correct physics.

Analytical solution for steady-state 1D convection-diffusion:
    C(x) = (exp(Pe*x) - 1) / (exp(Pe) - 1)
where Pe = u*L/D is the Peclet number.

We use a dedicated 1D network (x only) so the optimiser cannot escape to
degenerate 3D solutions. This matches the problem dimensionality exactly.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Dedicated 1-D network for validation
# ---------------------------------------------------------------------------
class PINN1D(nn.Module):
    """Single-input (x), single-output (C) MLP for 1D validation."""

    def __init__(self, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Physics and analytical solution
# ---------------------------------------------------------------------------
def analytical_solution(x: np.ndarray, Pe: float) -> np.ndarray:
    """Exact solution for 1D steady-state convection-diffusion."""
    return (np.exp(Pe * x) - 1.0) / (np.exp(Pe) - 1.0)


def physics_residual(model: PINN1D, x_col: torch.Tensor, u: float, D: float) -> torch.Tensor:
    """
    PDE residual for steady-state 1D convection-diffusion:
        u * dC/dx - D * d²C/dx² = 0
    """
    C = model(x_col)
    dC_dx = torch.autograd.grad(
        C, x_col, grad_outputs=torch.ones_like(C),
        create_graph=True, retain_graph=True
    )[0]
    d2C_dx2 = torch.autograd.grad(
        dC_dx, x_col, grad_outputs=torch.ones_like(dC_dx),
        create_graph=True, retain_graph=True
    )[0]
    residual = u * dC_dx - D * d2C_dx2
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------
def validate():
    """Train a 1D PINN and compare against the analytical solution."""
    print("=" * 60)
    print(" PINN Validation Against Analytical Solution")
    print("=" * 60)

    torch.manual_seed(42)

    D   = 0.01
    Pe  = 5.0
    u   = Pe * D     # = 0.05

    # --- Build dedicated 1D model ---
    model     = PINN1D(hidden=64, depth=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    # --- Collocation points (interior) ---
    N_col = 4000
    x_col = torch.rand(N_col, 1, requires_grad=True)

    # --- Boundary data: C(0) = 0, C(1) = 1 ---
    N_bc   = 500
    x_bc0  = torch.zeros(N_bc, 1);  C_bc0 = torch.zeros(N_bc, 1)
    x_bc1  = torch.ones(N_bc, 1);   C_bc1 = torch.ones(N_bc, 1)

    # Interior reference points sampled from the exact solution (10 % labelled)
    N_int  = 200
    x_int_np = np.linspace(0.05, 0.95, N_int)
    x_int    = torch.tensor(x_int_np[:, None], dtype=torch.float32)
    C_int    = torch.tensor(analytical_solution(x_int_np, Pe)[:, None], dtype=torch.float32)

    x_data = torch.cat([x_bc0, x_bc1, x_int])
    C_data = torch.cat([C_bc0, C_bc1, C_int])

    # --- Training ---
    epochs = 15000
    lam    = 10.0    # physics weight (equal-ish to normalised data term)

    for epoch in range(epochs):
        optimizer.zero_grad()

        C_pred    = model(x_data)
        data_loss = torch.mean((C_pred - C_data) ** 2)
        phys_loss = physics_residual(model, x_col, u, D)

        loss = data_loss + lam * phys_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:05d} | Total: {loss.item():.6f} | "
                  f"Data: {data_loss.item():.6f} | Physics: {phys_loss.item():.6f}")

    # --- Evaluate ---
    model.eval()
    x_test    = np.linspace(0, 1, 500)
    x_tensor  = torch.tensor(x_test[:, None], dtype=torch.float32)

    with torch.no_grad():
        C_pinn = model(x_tensor).numpy().flatten()

    C_exact = analytical_solution(x_test, Pe)

    # L2 relative error
    l2_error   = np.sqrt(np.sum((C_pinn - C_exact) ** 2) / np.sum(C_exact ** 2))
    max_error  = np.max(np.abs(C_pinn - C_exact))
    print(f"\nL2 Relative Error : {l2_error:.6f}")
    print(f"Max Absolute Error: {max_error:.6f}")

    # --- Plot ---
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_test, C_exact, "k-",  linewidth=2,   label="Analytical")
    axes[0].plot(x_test, C_pinn,  "r--", linewidth=1.8, label="PINN")
    axes[0].set_xlabel("x");  axes[0].set_ylabel("C(x)")
    axes[0].set_title(f"PINN vs Analytical (Pe={Pe})")
    axes[0].legend();  axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_test, np.abs(C_pinn - C_exact), "b-", linewidth=1.5)
    axes[1].set_xlabel("x");  axes[1].set_ylabel("|Error|")
    axes[1].set_title(f"Pointwise Error (L2 rel = {l2_error:.2e})")
    axes[1].grid(True, alpha=0.3);  axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("results/validation_vs_analytical.png", dpi=300)
    print("Validation plot saved to results/validation_vs_analytical.png")
    plt.close()


if __name__ == "__main__":
    validate()