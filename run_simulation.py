import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.train import train_pinn

def visualize_2d_results(model, title_suffix, save_path):
    """
    Evaluates the 2D model at time t=1.0 and plots a spatial cross-section.
    """
    model.eval()
    
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32)
    
    # Take a snapshot exactly at t = 1.0 (after fluid has flowed in)
    t_flat = torch.ones_like(x_flat)
    
    with torch.no_grad():
        C_pred = model(x_flat, y_flat, t_flat).numpy()
        
    C_grid = C_pred.reshape(X.shape)
    
    plt.figure(figsize=(10, 4))
    # We use 'turbo' colormap as it excels at showing fluid dynamics
    contour = plt.contourf(X, Y, C_grid, 60, cmap='turbo')
    plt.colorbar(contour, label='Toxin Concentration $C(x,y)$ at t=1')
    
    plt.xlabel('Length along Capillary (x)')
    plt.ylabel('Width across Capillary (y)')
    plt.title(f'2D Glomerular Flow Profile\n{title_suffix}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print("="*50)
    print(" 2D PINN: Glomerular Filtration Simulation")
    print("="*50)
    
    os.makedirs('results', exist_ok=True)
    
    # Healthy state simulation
    print("\n[+] Training Normotensive State Model...")
    model_healthy = train_pinn(epochs=6000, lr=1e-3, u_max=1.0, D=0.01, k=1.5)
    visualize_2d_results(model_healthy, 
                         title_suffix="Normotensive State (Effective Solute Clearance)",
                         save_path="results/healthy_kidney_2D.png")
                      
    # Diseased state simulation
    print("\n[+] Training Hypertensive State Model...")
    model_diseased = train_pinn(epochs=6000, lr=1e-3, u_max=3.5, D=0.01, k=0.1)
    visualize_2d_results(model_diseased, 
                         title_suffix="Hypertensive State (Impaired Filtration)",
                         save_path="results/diseased_kidney_2D.png")
                         
    print("\n[+] Simulations complete. Output saved to 'results/' directory.")

 