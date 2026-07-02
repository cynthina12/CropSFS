import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import allel


def process_vcf(vcf_path, save_path):
    print("=" * 60)
    print(f"[*] Path: {vcf_path}")
    
    if not os.path.exists(vcf_path):
        print(f"[X] Error: File not found at '{vcf_path}'")
        sys.exit(1)

    start_time = time.time()

    vcf_data = allel.read_vcf(vcf_path, fields=['samples', 'variants/CHROM', 'variants/POS', 'calldata/GT'])
    
    samples = vcf_data['samples']  
    chroms = vcf_data['variants/CHROM'].astype(str)
    positions = vcf_data['variants/POS'].astype(str)

    variant_ids = chroms + '_' + positions
    print(f"[+] Variants: {len(variant_ids)} | Samples: {len(samples)}")

    gt = vcf_data['calldata/GT']  
    matrix = gt.sum(axis=2)

    if (matrix < 0).any():
        missing_count = (matrix < 0).sum()
        print(f"[X] Critical Error: Detected {missing_count} missing entries in VCF.")
        print("[X] Execution terminated.")
        sys.exit(1)

    matrix = matrix.astype(np.int8)

    df_base = pd.DataFrame(
        data=matrix,
        index=variant_ids,
        columns=samples
    )
    
    df_final = df_base.T
    print(f"[+] Transposed Shape: {df_final.shape}")
    print("[✓] QC passed.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_final.to_pickle(save_path)

    duration = time.time() - start_time
    print("=" * 60)
    print(f"[✓] Success: {duration:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf_path", type=str, default="data/example.vcf.gz")
    parser.add_argument("--save_path", type=str, default="data/final1.pkl")
    args = parser.parse_args()

    process_vcf(vcf_path=args.vcf_path, save_path=args.save_path)


if __name__ == '__main__':
    main()
