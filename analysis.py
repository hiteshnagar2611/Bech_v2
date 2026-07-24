import csv
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_FILES = {
    'ESM1b': 'results/esm1b_results.tsv',
    'ESM2': 'results/esm2_results.tsv',
    'ESM3': 'results/esm3_results.tsv',
    'ProtT5': 'results/prott5_results.tsv',
    'ESM1b-MLM': 'results/esm1b_mlm_results.tsv',
    'DNABERT-2': 'results/dnabert2_results.tsv',
    'HyenaDNA': 'results/hyenadna_results.tsv',
    'NT-v2': 'results/ntv2_results.tsv',
}

COLORS = {
    'ESM1b': '#1f77b4',
    'ESM2': '#ff7f0e',
    'ESM3': '#2ca02c',
    'ProtT5': '#d62728',
    'ESM1b-MLM': '#9467bd',
    'DNABERT-2': '#e377c2',
    'HyenaDNA': '#8c564b',
    'NT-v2': '#7f7f7f',
}


def load_results(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df['label_binary'] = (df['label'] == 'Pathogenic').astype(int)
    return df


def compute_metrics(df):
    y_true = df['label_binary'].values
    y_score = df['cosine_distance'].values

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    optimal_threshold = thresholds_roc[opt_idx]

    y_pred = (y_score >= optimal_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true)

    path_dists = y_score[y_true == 1]
    ben_dists = y_score[y_true == 0]
    u_stat, p_val = mannwhitneyu(path_dists, ben_dists, alternative='two-sided')

    spearman_corr, spearman_p = spearmanr(y_score, y_true)

    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Optimal_Threshold': optimal_threshold,
        'MannWhitney_U': u_stat,
        'MannWhitney_p': p_val,
        'Spearman_rho': spearman_corr,
        'Spearman_p': spearman_p,
        'N_pathogenic': int(np.sum(y_true == 1)),
        'N_benign': int(np.sum(y_true == 0)),
    }


def plot_auroc_comparison(all_metrics, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(all_metrics.keys())
    aurocs = [all_metrics[m]['AUROC'] for m in models]
    colors = [COLORS.get(m, '#333') for m in models]

    bars = ax.bar(models, aurocs, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Zero-Shot Pathogenic Prediction: Cosine Distance\nAUROC Comparison', fontsize=13)
    ax.set_ylim(min(aurocs) - 0.05, 1.01)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auroc_comparison.png'), dpi=300)
    plt.close()


def plot_distance_distributions(all_dfs, output_dir):
    fig, axes = plt.subplots(1, len(all_dfs), figsize=(6 * len(all_dfs), 5), sharey=True)
    if len(all_dfs) == 1:
        axes = [axes]

    for ax, (model_name, df) in zip(axes, all_dfs.items()):
        path_dists = df[df['label'] == 'Pathogenic']['cosine_distance']
        ben_dists = df[df['label'] == 'Benign']['cosine_distance']

        ax.hist(ben_dists, bins=50, alpha=0.6, color='steelblue', density=True, label='Benign')
        ax.hist(path_dists, bins=50, alpha=0.6, color='crimson', density=True, label='Pathogenic')
        ax.set_title(model_name, fontsize=13)
        ax.set_xlabel('Cosine Distance', fontsize=11)
        ax.legend(fontsize=10)
        sns.despine(ax=ax)

    axes[0].set_ylabel('Density', fontsize=11)
    plt.suptitle('Cosine Distance Distributions: Pathogenic vs Benign', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curves(all_dfs, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, df in all_dfs.items():
        y_true = df['label_binary'].values
        y_score = df['cosine_distance'].values
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=COLORS.get(model_name, '#333'),
                label=f'{model_name} (AUPRC={auprc:.4f})', linewidth=2)

    baseline = all_dfs[list(all_dfs.keys())[0]]['label_binary'].mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=13)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=300)
    plt.close()


def plot_boxplot(all_dfs, output_dir):
    frames = []
    for model_name, df in all_dfs.items():
        tmp = df[['cosine_distance', 'label']].copy()
        tmp['model'] = model_name
        frames.append(tmp)
    combined = pd.concat(frames)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=combined, x='model', y='cosine_distance', hue='label',
                palette={'Pathogenic': 'crimson', 'Benign': 'steelblue'},
                fliersize=1, ax=ax)
    ax.set_title('Cosine Distance by Model and Class', fontsize=13)
    ax.set_ylabel('Cosine Distance', fontsize=12)
    ax.set_xlabel('')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_by_class.png'), dpi=300)
    plt.close()


def plot_roc_curves(all_dfs, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, df in all_dfs.items():
        y_true = df['label_binary'].values
        y_score = df['cosine_distance'].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=COLORS.get(model_name, '#333'),
                label=f'{model_name} (AUROC={auroc:.4f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=13)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
    plt.close()


def per_gene_analysis(all_dfs, output_dir, top_n=10):
    summary_frames = []
    for model_name, df in all_dfs.items():
        gene_counts = df.groupby('gene_symbol').size().sort_values(ascending=False).head(top_n)
        for gene in gene_counts.index:
            gene_df = df[df['gene_symbol'] == gene]
            if gene_df['label_binary'].nunique() < 2:
                continue
            try:
                auroc = roc_auc_score(gene_df['label_binary'], gene_df['cosine_distance'])
            except ValueError:
                auroc = np.nan
            summary_frames.append({
                'Model': model_name,
                'Gene': gene,
                'N_variants': len(gene_df),
                'AUROC': auroc,
            })

    if summary_frames:
        per_gene_df = pd.DataFrame(summary_frames)
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = per_gene_df.pivot(index='Gene', columns='Model', values='AUROC')
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'AUROC by Gene (Top {top_n} genes by variant count)', fontsize=13)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'auroc_per_gene.png'), dpi=300)
        plt.close()
        per_gene_df.to_csv(os.path.join(output_dir, 'per_gene_auroc.csv'), index=False)
        print(f"Per-gene AUROC: {os.path.join(output_dir, 'per_gene_auroc.csv')}")


def main():
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = {}
    all_metrics = {}

    for model_name, filepath in MODEL_FILES.items():
        if not os.path.exists(filepath):
            print(f"Skipping {model_name}: {filepath} not found")
            continue
        df = load_results(filepath)
        all_dfs[model_name] = df
        metrics = compute_metrics(df)
        all_metrics[model_name] = metrics
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    if not all_dfs:
        print("No result files found! Run run_benchmark.py first.")
        return

    summary_df = pd.DataFrame(all_metrics).T
    summary_df.to_csv(os.path.join(output_dir, 'benchmark_summary.csv'))
    print(f"\nSummary: {os.path.join(output_dir, 'benchmark_summary.csv')}")

    plot_auroc_comparison(all_metrics, output_dir)
    plot_distance_distributions(all_dfs, output_dir)
    plot_roc_curves(all_dfs, output_dir)
    plot_pr_curves(all_dfs, output_dir)
    plot_boxplot(all_dfs, output_dir)
    per_gene_analysis(all_dfs, output_dir)

    print(f"\nAll plots saved to {output_dir}/")
    print("Files:")
    for f in os.listdir(output_dir):
        if f.endswith('.png') or f.endswith('.csv'):
            print(f"  {f}")


if __name__ == '__main__':
    main()
