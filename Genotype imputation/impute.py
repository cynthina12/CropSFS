import argparse
import os
import sys
import time
import glob
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import numpy._core.numeric as _num
except ModuleNotFoundError:
    import numpy.core.numeric as _num
    sys.modules['numpy._core.numeric'] = _num

NUM_IN = 4096
NUM_CLS = 3
MISS_IDX = 4
NUM_ALLELES = 5
TASK_WIDTH = 100


class CrossAttn(nn.Module):
    def __init__(self, d, h=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, q, kv):
        h, _ = self.attn(q, kv, kv, need_weights=False)
        h = self.ln1(h + q)
        return self.ln2(self.ff(h) + h)


class LatentBlock(nn.Module):
    def __init__(self, d, h=8):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        h, _ = self.sa(x, x, x, need_weights=False)
        h = self.ln1(h + x)
        return self.ln2(self.ff(h) + h)


class PerceiverMultiTask(nn.Module):
    def __init__(self, num_tasks, task_width, d_model=256, latent_dim=512, n_blocks=4, n_heads=8):
        super().__init__()
        self.num_tasks = num_tasks
        self.in_emb = nn.Embedding(NUM_ALLELES, d_model, padding_idx=MISS_IDX)
        self.in_pos = nn.Embedding(NUM_IN, d_model)
        self.latent = nn.Parameter(torch.randn(latent_dim, d_model) * 0.02)
        self.xin = CrossAttn(d_model, n_heads)
        self.blocks = nn.ModuleList([LatentBlock(d_model, n_heads) for _ in range(n_blocks)])
        self.xout = CrossAttn(d_model, n_heads)
        self.task_query = nn.Parameter(torch.randn(num_tasks, task_width, d_model) * 0.02)
        self.task_heads = nn.ModuleList([nn.Linear(d_model, NUM_CLS) for _ in range(num_tasks)])

    def forward(self, x):
        B = x.size(0)
        inp = self.in_emb(x) + self.in_pos.weight[None]
        lat = self.latent[None].expand(B, -1, -1)
        lat = self.xin(lat, inp)
        for blk in self.blocks:
            lat = blk(lat)
        out_list = []
        for t in range(self.num_tasks):
            q = self.task_query[t][None].expand(B, -1, -1)
            out = self.xout(q, lat)
            out_list.append(self.task_heads[t](out))
        return out_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/CropSFS4096.pkl")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--output_path", type=str, default="output/imputed_full_genome.pkl")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target device: {device}")

    if not os.path.exists(args.input_path):
        print(f"[X] Error: Input file not found at '{args.input_path}'")
        sys.exit(1)

    print(f"[*] Loading low-density markers from: {args.input_path}")
    sl_4096 = pd.read_pickle(args.input_path)
    samples = sl_4096.index
    
    x_arr = sl_4096.fillna(MISS_IDX).astype(np.int64).values
    x_tensor = torch.as_tensor(x_arr, dtype=torch.long)

    ckpt_files = glob.glob(os.path.join(args.weights_dir, "seg*.pt"))
    if not ckpt_files:
        print(f"[X] Error: No checkpoints found in '{args.weights_dir}'")
        sys.exit(1)

    def extract_seg_idx(filename):
        match = re.search(r"seg(\d+)", os.path.basename(filename))
        return int(match.group(1)) if match else -1

    ckpt_files = sorted(ckpt_files, key=extract_seg_idx)
    print(f"[+] Found {len(ckpt_files)} trained segments to process.")

    segment_outputs = []
    
    for ckpt_path in ckpt_files:
        t0 = time.time()
        tag = os.path.splitext(os.path.basename(ckpt_path))[0]
        
        meta = torch.load(ckpt_path, map_location="cpu")
        num_tasks = meta["num_tasks"]
        task_slices = meta["task_slices"]
        seg_len = meta["seg_len"]
        actual_task_widths = [e - s for s, e in task_slices]
        max_task_width = max(actual_task_widths)

        model = PerceiverMultiTask(
            num_tasks=num_tasks, task_width=max_task_width,
            d_model=256, latent_dim=512, n_blocks=4, n_heads=8,
        ).to(device)
        
        model.load_state_dict({k: v.to(device) for k, v in meta["model_state"].items()})
        model.eval()

        seg_preds = []
        with torch.no_grad():
            for i in range(0, len(x_tensor), args.batch_size):
                batch_x = x_tensor[i:i+args.batch_size].to(device)
                logits = model(batch_x)
                
                bp = np.concatenate([
                    logits[t][:, :actual_task_widths[t], :].argmax(-1).cpu().numpy() 
                    for t in range(num_tasks)
                ], axis=1)
                seg_preds.append(bp)
                
        seg_matrix = np.concatenate(seg_preds, axis=0)
        segment_outputs.append(seg_matrix)
        
        print(f"[-] Completed: {tag} | shape: {seg_matrix.shape} | time: {time.time() - t0:.2f}s")

    print("[*] Merging all segments into full genome...")
    full_genome_matrix = np.concatenate(segment_outputs, axis=1)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_imputed = pd.DataFrame(data=full_genome_matrix, index=samples).astype(np.int8)
    df_imputed.to_pickle(args.output_path)
    
    print(f"[✓] Success: Full genome imputed matrix saved to -> {args.output_path}")
    print(f"[✓] Final Shape: {df_imputed.shape}")


if __name__ == "__main__":
    main()
