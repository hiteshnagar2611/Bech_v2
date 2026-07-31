"""
Gene-Disjoint Train/Test Split
Ensures NO gene appears in both train and test sets.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def create_gene_disjoint_split(dataset_dir, test_size=0.2, random_state=42):
    feat_path = os.path.join(dataset_dir, 'feature_matrix.tsv')
    df = pd.read_csv(feat_path, sep='\t')
    print(f"Loaded: {df.shape[0]} variants, {df['gene_symbol'].nunique()} genes")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    gene_stats = df.groupby('gene_symbol').agg(
        n_variants=('label_binary', 'count'),
        n_pathogenic=('label_binary', 'sum'),
    ).reset_index()
    gene_stats['n_benign'] = gene_stats['n_variants'] - gene_stats['n_pathogenic']

    print(f"\nGene variant count distribution:")
    print(f"  Min: {gene_stats['n_variants'].min()}")
    print(f"  Max: {gene_stats['n_variants'].max()}")
    print(f"  Median: {gene_stats['n_variants'].median():.0f}")
    print(f"  Mean: {gene_stats['n_variants'].mean():.1f}")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df['gene_symbol']))

    train_genes = set(df.iloc[train_idx]['gene_symbol'].unique())
    test_genes = set(df.iloc[test_idx]['gene_symbol'].unique())
    overlap = train_genes & test_genes
    print(f"\n=== Gene-Disjoint Split ===")
    print(f"Train genes: {len(train_genes)}")
    print(f"Test genes: {len(test_genes)}")
    print(f"Gene overlap: {len(overlap)} (should be 0)")
    assert len(overlap) == 0, f"Gene overlap detected: {overlap}"

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print(f"\nTrain: {len(train_df)} variants")
    print(f"  Pathogenic: {int(train_df['label_binary'].sum())}")
    print(f"  Benign: {int((train_df['label_binary']==0).sum())}")
    print(f"  Ratio: {train_df['label_binary'].mean():.3f}")

    print(f"\nTest: {len(test_df)} variants")
    print(f"  Pathogenic: {int(test_df['label_binary'].sum())}")
    print(f"  Benign: {int((test_df['label_binary']==0).sum())}")
    print(f"  Ratio: {test_df['label_binary'].mean():.3f}")

    train_pathogenic_genes = set(train_df[train_df['label_binary']==1]['gene_symbol'].unique())
    test_pathogenic_genes = set(test_df[test_df['label_binary']==1]['gene_symbol'].unique())
    print(f"\nPathogenic genes: train={len(train_pathogenic_genes)}, test={len(test_pathogenic_genes)}")
    print(f"Pathogenic gene overlap: {len(train_pathogenic_genes & test_pathogenic_genes)}")

    split_dir = os.path.join(dataset_dir, 'splits')
    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, 'gene_disjoint_train.tsv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(split_dir, 'gene_disjoint_test.tsv'), sep='\t', index=False)

    stats = {
        'train_genes': len(train_genes),
        'test_genes': len(test_genes),
        'gene_overlap': len(overlap),
        'train_variants': len(train_df),
        'test_variants': len(test_df),
        'train_pathogenic': int(train_df['label_binary'].sum()),
        'train_benign': int((train_df['label_binary']==0).sum()),
        'test_pathogenic': int(test_df['label_binary'].sum()),
        'test_benign': int((test_df['label_binary']==0).sum()),
    }
    with open(os.path.join(split_dir, 'split_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {split_dir}/")
    return train_df, test_df


if __name__ == '__main__':
    ds_dir = sys.argv[1] if len(sys.argv) > 1 else 'dataset1_clinvar_only'
    create_gene_disjoint_split(ds_dir)
