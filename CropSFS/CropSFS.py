import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import numpy._core.numeric as _num
except ModuleNotFoundError:
    import numpy.core.numeric as _num
    sys.modules['numpy._core.numeric'] = _num


class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(SimpleDiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        t_embed = torch.ones(x.size(0), 1, device=x.device) * t
        x_t = torch.cat([x, t_embed], dim=1)
        return self.net(x_t)


def train_diffusion_model(data, device, param_grid=None, patience=20, max_epochs=1000, batch_size=64):
    if param_grid is None:
        param_grid = {'lr': [0.001], 'hidden_dim': [256]}

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    
    best_loss = float('inf')
    best_model = None
    best_params = None
    global_start_time = time.time()

    for params in ParameterGrid(param_grid):
        param_start_time = time.time()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        model = SimpleDiffusionModel(input_dim=data.shape[1], hidden_dim=params['hidden_dim']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        no_improve_count = 0
        prev_loss = float('inf')

        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_x in dataloader:
                batch_x = batch_x[0].to(device)
                t = torch.rand(1).item()
                noisy_data = batch_x + torch.randn_like(batch_x) * t
                
                with torch.amp.autocast(device_type=device.type):
                    pred_noise = model(noisy_data, t)
                    loss = ((pred_noise - (noisy_data - batch_x))**2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_x.size(0)
            
            total_epoch_loss = epoch_loss / len(data_tensor)

            if total_epoch_loss < prev_loss - 1e-4:
                no_improve_count = 0
                prev_loss = total_epoch_loss
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

        param_duration = time.time() - param_start_time
        print(f"[-] Params: {params} | Loss: {prev_loss:.6f} | Time: {param_duration:.2f}s")

        if prev_loss < best_loss:
            best_loss = prev_loss
            best_model = SimpleDiffusionModel(input_dim=data.shape[1], hidden_dim=params['hidden_dim'])
            best_model.load_state_dict(model.state_dict())
            best_params = params

    global_duration = time.time() - global_start_time
    print('\n' + '='*50)
    print(f"[+] Best: {best_params} | Loss: {best_loss:.6f}")
    print(f"[+] Search time: {global_duration/60:.2f} mins")
    print('='*50 + '\n')
    
    return best_model.to(device), scaler


def get_embeddings(model, data, scaler, device, t=0.5, batch_size=128):
    data_scaled = scaler.transform(data)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for batch_x in dataloader:
            batch_x = batch_x[0].to(device)
            t_embed = torch.ones(batch_x.size(0), 1, device=device) * t
            emb = model.net[:-1](torch.cat([batch_x, t_embed], dim=1))
            embeddings_list.append(emb.cpu().numpy())
            
    return np.vstack(embeddings_list)


def laplacian_scores(data, embeddings, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
    W = nbrs.kneighbors_graph(embeddings, mode='connectivity').toarray()

    D = np.diag(W.sum(axis=1))
    L = D - W

    scores = []
    for i in range(data.shape[1]):
        f = data.iloc[:, i].values
        numerator = f.T @ L @ f
        denominator = f.T @ D @ f + 1e-8
        scores.append(numerator / denominator)

    return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/final1.pkl")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--batch_size", type=str, default="64")
    parser.add_argument("--counts", type=str, default="4000")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Env -> PyTorch: {torch.__version__} | Device: {device}")

    print(f"[*] Loading: {args.input_path}")
    if not os.path.exists(args.input_path):
        print(f"[X] Error: File not found at '{args.input_path}'")
        sys.exit(1)
        
    df = pd.read_pickle(args.input_path).astype('int8')
    os.makedirs(args.output_dir, exist_ok=True)

    param_grid = {
        'lr': [0.01, 0.001, 0.0001], 
        'hidden_dim': [128, 256, 512]
    } 
    
    model, scaler = train_diffusion_model(df, device=device, param_grid=param_grid, batch_size=int(args.batch_size))
    
    print("[*] Extracting embeddings...")
    embeddings = get_embeddings(model, df, scaler, device=device, batch_size=int(args.batch_size))
    
    print("[*] Calculating Laplacian scores...")
    scores = laplacian_scores(df, embeddings)
    feature_ranking = np.argsort(scores)
    
    target_counts = [int(n.strip()) for n in args.counts.split(",") if n.strip().isdigit()]
    
    for n in target_counts:
        current_features = feature_ranking[:n]
        selected_df = df.iloc[:, current_features]
        
        save_path = os.path.join(args.output_dir, f"final_selected_{n}.csv")
        selected_df.to_csv(save_path)
        print(f"[✓] Saved: {save_path} | Shape: {selected_df.shape}")

    print("[*] Done.")


if __name__ == "__main__":
    main()
