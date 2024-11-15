"""
Parameter Sensitivity Study for the 2D PINN Glomerular Filtration Model.

Sweeps across key physical parameters (u_max, k) to characterize how
filtration performance degrades under varying hemodynamic conditions.
Produces a grid of concentration profiles and a summary heatmap.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.train import train_pinn


def evaluate_model(model, title, save_path):
    """Evaluate trained model at t=1.0 and save concentration plot."""
    model.eval()
    x = np.linspace(0, 1, 80)
    y = np.linspace(0, 1, 80)
    X, Y = np.meshgrid(x, y)

    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32)
    t_flat = torch.ones_like(x_flat)

    with torch.no_grad():
        C_pred = model(x_flat, y_flat, t_flat).numpy()

    C_grid = C_pred.reshape(X.shape)

    plt.figure(figsize=(8, 3.5))
    contour = plt.contourf(X, Y, C_grid, 50, cmap='turbo')
    plt.colorbar(contour, label='C(x,y) at t=1')
    plt.xlabel('Capillary Length (x)')
    plt.ylabel('Capillary Width (y)')
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return C_grid


def compute_outlet_retention(C_grid):
    """Compute mean concentration at the outlet (x=1) as a filtration metric."""
    return float(np.mean(C_grid[:, -1]))


if __name__ == "__main__":
    print("=" * 60)
    print(" Parameter Sensitivity Study: u_max vs k")
    print("=" * 60)

    os.makedirs('results/sensitivity', exist_ok=True)

    # Parameter grid
    u_max_values = [0.5, 1.0, 2.0, 3.5]
    k_values = [0.1, 0.5, 1.0, 1.5, 2.0]

    retention_matrix = np.zeros((len(u_max_values), len(k_values)))

    for i, u_max in enumerate(u_max_values):
        for j, k in enumerate(k_values):
            label = f"u_max={u_max}, k={k}"
            print(f"\n[+] Training: {label}")

            model = train_pinn(epochs=3000, lr=1e-3, u_max=u_max, D=0.01, k=k)
            C_grid = evaluate_model(
                model,
                title=f"$u_{{max}}$={u_max}, $k$={k}",
                save_path=f"results/sensitivity/umax{u_max}_k{k}.png"
            )
            retention_matrix[i, j] = compute_outlet_retention(C_grid)
            print(f"    Outlet retention: {retention_matrix[i, j]:.4f}")

    # Summary heatmap
    plt.figure(figsize=(8, 5))
    plt.imshow(retention_matrix, cmap='hot', aspect='auto', origin='lower')
    plt.colorbar(label='Mean Outlet Concentration (Retention)')
    plt.xticks(range(len(k_values)), [str(k) for k in k_values])
    plt.yticks(range(len(u_max_values)), [str(u) for u in u_max_values])
    plt.xlabel('Filtration Rate (k)')
    plt.ylabel('Max Velocity ($u_{max}$)')
    plt.title('Parameter Sensitivity: Solute Retention at Capillary Outlet')
    plt.tight_layout()
    plt.savefig('results/sensitivity/retention_heatmap.png', dpi=300)
    print("\n[+] Heatmap saved to results/sensitivity/retention_heatmap.png")
    plt.close()

    print("\n[+] Sensitivity study complete.")
 