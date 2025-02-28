"""
Animated GIFs showing the temporal evolution of solute
concentration for Healthy vs Diseased kidney states.

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from src.train import train_pinn


def create_animation(model, title, save_path, fps=10):
    """
    Animate the 2D concentration field from t=0 to t=1.
    """
    model.eval()
    x = np.linspace(0, 1, 80)
    y = np.linspace(0, 1, 80)
    X, Y = np.meshgrid(x, y)
    t_values = np.linspace(0.01, 1.0, 40)

    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32)

    fig, ax = plt.subplots(figsize=(9, 4))

    # Pre-compute all frames
    frames = []
    for t_val in t_values:
        t_flat = torch.ones_like(x_flat) * t_val
        with torch.no_grad():
            C_pred = model(x_flat, y_flat, t_flat).numpy()
        frames.append(C_pred.reshape(X.shape))

    # Determine global color scale
    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)

    contour = ax.contourf(X, Y, frames[0], 50, cmap='turbo', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(contour, ax=ax, label='Concentration C(x,y)')
    ax.set_xlabel('Capillary Length (x)')
    ax.set_ylabel('Capillary Width (y)')
    time_text = ax.set_title(f'{title}\nt = {t_values[0]:.2f}', fontsize=10)

    def update(frame_idx):
        for c in ax.collections:
            c.remove()
        ax.contourf(X, Y, frames[frame_idx], 50, cmap='turbo', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Capillary Length (x)')
        ax.set_ylabel('Capillary Width (y)')
        ax.set_title(f'{title}\nt = {t_values[frame_idx]:.2f}', fontsize=10)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(t_values), interval=1000//fps, blit=False)
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Animation saved to {save_path}")


if __name__ == "__main__":
    print(" Animated GIFs: Healthy vs Diseased Filtration")
    
    os.makedirs('results/animations', exist_ok=True)

    # Train healthy model
    print("\nTraining Normotensive (Healthy) Model...")
    model_healthy = train_pinn(epochs=6000, lr=1e-3, u_max=1.0, D=0.01, k=1.5)
    create_animation(
        model_healthy,
        title="Normotensive State (Effective Clearance)",
        save_path="results/animations/healthy_filtration.gif"
    )

    # Train diseased model
    print("\nraining Hypertensive (Diseased) Model...")
    model_diseased = train_pinn(epochs=6000, lr=1e-3, u_max=3.5, D=0.01, k=0.1)
    create_animation(
        model_diseased,
        title="Hypertensive State (Impaired Filtration)",
        save_path="results/animations/diseased_filtration.gif"
    )

    print("\nAnimations complete. Check results/animations/")
