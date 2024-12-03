"""
Validation of the PINN against an analytical solution.

Uses the 1D steady-state convection-diffusion equation (no reaction term)
with known analytical solution to verify the PINN learns the correct physics.

Analytical solution for steady-state 1D convection-diffusion:
    C(x) = (exp(Pe*x) - 1) / (exp(Pe) - 1)
where Pe = u*L/D is the Peclet number.

We train the PINN on this simplified problem and compute the L2 relative error
against the exact solution.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.network import PINN


def analytical_solution(x, Pe):
    """Exact solution for 1D steady-state convection-diffusion."""
    return (np.exp(Pe * x) - 1.0) / (np.exp(Pe) - 1.0)


def compute_1d_physics_loss(model, x_col, Pe, D):
    """
    1D steady-state convection-diffusion PDE residual:
    u * dC/dx - D * d²C/dx² = 0
    """
    C = model(x_col, torch.full_like(x_col, 0.5), torch.ones_like(x_col))

    dC_dx = torch.autograd.grad(C, x_col, grad_outputs=torch.ones_like(C),
                                 create_graph=True, retain_graph=True)[0]
    d2C_dx2 = torch.autograd.grad(dC_dx, x_col, grad_outputs=torch.ones_like(dC_dx),
                                   create_graph=True, retain_graph=True)[0]

    u = Pe * D  # velocity from Peclet number
    residual = u * dC_dx - D * d2C_dx2
    return torch.mean(residual ** 2)


def validate():
    """Train a simplified PINN and compare to analytical solution."""
    print("=" * 60)
    print(" PINN Validation Against Analytical Solution")
    print("=" * 60)

    D = 0.01
    Pe = 5.0  # Peclet number
    u = Pe * D

    # Build model
    model = PINN(layers=[3, 40, 40, 40, 40, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Collocation points
    N_col = 2000
    x_col = torch.rand(N_col, 1, requires_grad=True)

    # Boundary data: C(0) = 0, C(1) = 1
    N_bc = 200
    x_bc0 = torch.zeros(N_bc, 1)
    C_bc0 = torch.zeros(N_bc, 1)
    x_bc1 = torch.ones(N_bc, 1)
    C_bc1 = torch.ones(N_bc, 1)

    x_data = torch.cat([x_bc0, x_bc1])
    C_data = torch.cat([C_bc0, C_bc1])

    epochs = 4000
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Data loss (boundary conditions)
        C_pred = model(x_data, torch.full_like(x_data, 0.5), torch.ones_like(x_data))
        data_loss = torch.mean((C_pred - C_data) ** 2)

        # Physics loss
        physics_loss = compute_1d_physics_loss(model, x_col, Pe, D)

        total_loss = 200.0 * data_loss + physics_loss
        total_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:05d} | Total: {total_loss.item():.6f} | "
                  f"Data: {data_loss.item():.6f} | Physics: {physics_loss.item():.6f}")

    # Evaluate and compare
    model.eval()
    x_test = np.linspace(0, 1, 200)
    x_tensor = torch.tensor(x_test[:, None], dtype=torch.float32)

    with torch.no_grad():
        C_pinn = model(x_tensor, torch.full_like(x_tensor, 0.5),
                       torch.ones_like(x_tensor)).numpy().flatten()

    C_exact = analytical_solution(x_test, Pe)

    # L2 relative error
    l2_error = np.sqrt(np.sum((C_pinn - C_exact) ** 2) / np.sum(C_exact ** 2))
    print(f"\nL2 Relative Error: {l2_error:.6f}")
    print(f"Max Absolute Error: {np.max(np.abs(C_pinn - C_exact)):.6f}")

    # Plot comparison
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_test, C_exact, 'k-', linewidth=2, label='Analytical')
    plt.plot(x_test, C_pinn, 'r--', linewidth=2, label='PINN')
    plt.xlabel('x')
    plt.ylabel('C(x)')
    plt.title(f'PINN vs Analytical (Pe={Pe})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x_test, np.abs(C_pinn - C_exact), 'b-', linewidth=1.5)
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.title(f'Pointwise Error (L2 rel = {l2_error:.2e})')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('results/validation_vs_analytical.png', dpi=300)
    print("Validation plot saved to results/validation_vs_analytical.png")
    plt.close()


if __name__ == "__main__":
    validate()
