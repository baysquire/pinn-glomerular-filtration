import torch
import torch.optim as optim
from .network import PINN
from .physics import compute_physics_loss

def train_pinn(epochs=6000, lr=1e-3, u_max=1.0, D=0.01, k=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training 2D PINN on device: {device}")
    
    # Deeper network for 2D complexity
    model = PINN(layers=[3, 40, 40, 40, 40, 1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # --- Domain Collocation Points ---
    N_collocation = 4000
    x_col = torch.rand(N_collocation, 1, requires_grad=True).to(device)
    y_col = torch.rand(N_collocation, 1, requires_grad=True).to(device)
    t_col = torch.rand(N_collocation, 1, requires_grad=True).to(device)
    
    N_bc = 800
    
    # 1. Initial Condition: t=0, kidney is completely empty
    x_ic = torch.rand(N_bc, 1).to(device)
    y_ic = torch.rand(N_bc, 1).to(device)
    t_ic = torch.zeros(N_bc, 1).to(device)
    C_ic = torch.zeros(N_bc, 1).to(device)
    
    # 2. Inlet Boundary: x=0, toxin constantly flows in
    x_inlet = torch.zeros(N_bc, 1).to(device)
    y_inlet = torch.rand(N_bc, 1).to(device)
    t_inlet = torch.rand(N_bc, 1).to(device)
    C_inlet = torch.ones(N_bc, 1).to(device)
    
    # 3. Wall Boundaries: y=0 and y=1. 
    # Healthy capillary walls filter toxins instantly upon touch (C=0)
    x_wall1 = torch.rand(N_bc, 1).to(device)
    y_wall1 = torch.zeros(N_bc, 1).to(device)
    t_wall1 = torch.rand(N_bc, 1).to(device)
    C_wall1 = torch.zeros(N_bc, 1).to(device)
    
    x_wall2 = torch.rand(N_bc, 1).to(device)
    y_wall2 = torch.ones(N_bc, 1).to(device)
    t_wall2 = torch.rand(N_bc, 1).to(device)
    C_wall2 = torch.zeros(N_bc, 1).to(device)
    
    # Combine all boundary data
    x_data = torch.cat([x_ic, x_inlet, x_wall1, x_wall2], dim=0)
    y_data = torch.cat([y_ic, y_inlet, y_wall1, y_wall2], dim=0)
    t_data = torch.cat([t_ic, t_inlet, t_wall1, t_wall2], dim=0)
    C_data_exact = torch.cat([C_ic, C_inlet, C_wall1, C_wall2], dim=0)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        C_data_pred = model(x_data, y_data, t_data)
        data_loss = torch.mean((C_data_pred - C_data_exact) ** 2)
        
        physics_loss = compute_physics_loss(model, x_col, y_col, t_col, u_max=u_max, D=D, k=k)
        
        # Heavy penalty weight to strictly enforce wall boundaries
        total_loss = (200.0 * data_loss) + physics_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:05d} | Total: {total_loss.item():.4f} | Data: {data_loss.item():.4f} | Physics: {physics_loss.item():.4f}")
            
    return model
       