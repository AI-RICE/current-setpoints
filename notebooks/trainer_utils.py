import torch
import copy
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device, min_delta=0.0, verbose=False):
    """
    Handles the PyTorch training and validation loop with early stopping.
    Receives dynamic B_tensors directly from the dataloader batches.
    """
    print(f"\nStarting Training with Early Stopping (Patience={patience})...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = copy.deepcopy(model.state_dict())  

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        # Unpack all 3 elements: inputs, targets, and dynamic B vectors
        for inputs, targets, b_tensors in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Reshape b_tensors to [Batch, Features, 1] to match neural.py expectations
            b_tensors = b_tensors.to(device).unsqueeze(-1)
            
            optimizer.zero_grad()
            outputs = model(inputs, b_tensors)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for inputs, targets, b_tensors in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                b_tensors = b_tensors.to(device).unsqueeze(-1)
                
                outputs = model(inputs, b_tensors)
                val_loss_sum += criterion(outputs, targets).item() * inputs.size(0)
        
        avg_val_loss = val_loss_sum / len(val_loader.dataset)

        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_weights = copy.deepcopy(model.state_dict())  
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                model.load_state_dict(best_weights)
                return best_val_loss, epoch + 1
                
    model.load_state_dict(best_weights)
    return best_val_loss, num_epochs


def evaluate_model(model, test_loader, criterion, device):
    """Evaluates the model on a test set and returns the RMSE."""
    model.eval()
    test_loss_sum = 0.0
    
    with torch.no_grad():
        for inputs, targets, b_tensors in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            b_tensors = b_tensors.to(device).unsqueeze(-1)
            
            outputs = model(inputs, b_tensors)
            test_loss_sum += criterion(outputs, targets).item() * inputs.size(0)
            
    return np.sqrt(test_loss_sum / len(test_loader.dataset))


def prepare_fold_dataloaders(X_train_np, X_val_np, y_train_np, y_val_np, B_train_np, B_val_np):
    """
    Fits the Scikit-Learn scaler strictly on the training fold, transforms both folds, 
    and returns PyTorch DataLoaders ready for training (now including B arrays).
    """
    scaler_X = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train_np)
    X_val_norm = scaler_X.transform(X_val_np)
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_norm).float(), 
        torch.from_numpy(y_train_np).float(),
        torch.from_numpy(B_train_np).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_norm).float(), 
        torch.from_numpy(y_val_np).float(),
        torch.from_numpy(B_val_np).float()
    )
    
    batch_size = min(64, len(X_train_norm) // 4) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler_X