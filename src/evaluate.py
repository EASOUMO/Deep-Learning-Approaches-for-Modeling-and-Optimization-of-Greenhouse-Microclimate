import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils import unscale_from_unit_range

def evaluate_model(model, data_loader, criterion, device='cpu', plot_path='evaluation_plot.png'):
    """
    Evaluates the model on the given data loader.
    Calculates MSE, MAE, RMSE, and R2 score.
    Plots Actual vs Predicted values.
    """
    model.eval()
    all_targets = []
    all_predictions = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            
    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    
    # Unscale data
    all_targets_unscaled = unscale_from_unit_range(all_targets)
    all_predictions_unscaled = unscale_from_unit_range(all_predictions)
    
    # Calculate Metrics on Unscaled Data
    mse = mean_squared_error(all_targets_unscaled, all_predictions_unscaled)
    mae = mean_absolute_error(all_targets_unscaled, all_predictions_unscaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets_unscaled, all_predictions_unscaled)
    
    avg_loss = total_loss / len(data_loader.dataset)
    
    metrics = {
        'loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    print("\n=== Model Evaluation Results ===")
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print("================================")
    
    # Plotting Unscaled Data
    num_targets = all_targets_unscaled.shape[1]
    num_points_to_plot = min(100, len(all_targets_unscaled))
    
    fig, axes = plt.subplots(num_targets, 1, figsize=(10, 6 * num_targets), sharex=True)
    if num_targets == 1:
        axes = [axes]
        
    target_names = ['u1', 'u2', 'u3'] if num_targets == 3 else [f'Target {i+1}' for i in range(num_targets)]
    
    for i in range(num_targets):
        axes[i].plot(all_targets_unscaled[:num_points_to_plot, i], label='Actual', marker='o')
        axes[i].plot(all_predictions_unscaled[:num_points_to_plot, i], label='Predicted', marker='x')
        axes[i].set_title(f'Actual vs Predicted ({target_names[i]})')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True)
        
    plt.xlabel('Sample Index')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Evaluation plot saved to {plot_path}")
    plt.close()
    
    return metrics
