import torch

def compute_physics_loss(model, x_col, y_col, t_col, u_max, D, k):
    """
    Computes the 2D Physics-Informed Loss.
    """
    C = model(x_col, y_col, t_col)
    
    # First derivatives
    dC_dt = torch.autograd.grad(C, t_col, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
    dC_dx = torch.autograd.grad(C, x_col, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
    dC_dy = torch.autograd.grad(C, y_col, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
    
    # Second derivatives (Diffusion in 2D)
    d2C_dx2 = torch.autograd.grad(dC_dx, x_col, grad_outputs=torch.ones_like(dC_dx), create_graph=True, retain_graph=True)[0]
    d2C_dy2 = torch.autograd.grad(dC_dy, y_col, grad_outputs=torch.ones_like(dC_dy), create_graph=True, retain_graph=True)[0]
    
    # UPGRADE: Parabolic velocity profile (Poiseuille Flow)
    # Fluid moves fastest in the center (y=0.5) and is stationary at the capillary walls (y=0, y=1)
    u_vel = u_max * (1.0 - ((y_col - 0.5) / 0.5)**2)
    
    # 2D Convection-Diffusion-Filtration Equation
    pde_residual = dC_dt + (u_vel * dC_dx) - D * (d2C_dx2 + d2C_dy2) + (k * C)
    
    return torch.mean(pde_residual ** 2)
  