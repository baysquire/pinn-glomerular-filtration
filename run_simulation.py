import torch
from src.train import train_pinn

if __name__ == "__main__":
    model = train_pinn(epochs=3000)  