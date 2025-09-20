from pathlib import Path
from time import time

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from python.utils.neural_network import NN

if __name__ == '__main__':
    input_path = Path('data/detection/train')
    export_path = Path('ml_models/')
    export_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path / 'dataset_v1.csv')
    df['n_std'] = df['dKe'] / df['std_surf']
    df = df.drop(['n_alpha', ], axis=1)

    X = df.drop('is_new', axis=1)
    y = df['is_new']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=69)

    # Standardize data
    scaler = StandardScaler()
    X_train_torch = scaler.fit_transform(X_train)
    X_test_torch = scaler.transform(X_test)
    joblib.dump(scaler, export_path / 'nn_scaler.pkl')

    X_train_torch = torch.tensor(X_train_torch, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test_torch, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Model and model params
    model = NN(X_train_torch.shape[1])

    batch_size = 2 ** 8
    lr = 1e-4

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    dataloader_train = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)

    epochs = 1000
    early_stop_at = 20

    print('\nTraining model')
    print(f'X_train len: {len(X_train)}')

    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        start_time = time()
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader_train:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(dataloader_train)
        scheduler.step(metrics=train_loss)
        end_time = time()
        print(
            f'Epoch {epoch + 1}: Loss: {train_loss:.4f}. Time: {end_time - start_time:.2f}s. Lr: {scheduler.get_last_lr()}')

        if epoch > 100:
            if train_loss < best_loss:
                best_loss = train_loss
            else:
                early_stop_counter += 1

        if early_stop_counter >= early_stop_at:
            print(f'Early stopping. Best loss: {best_loss:.4f}')
            break

    # Evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_torch)
        test_loss = criterion(test_outputs, y_test_torch)
        print(f"Test Loss: {test_loss.item()}")

    torch.save(model.state_dict(), export_path / 'nn.pth')
