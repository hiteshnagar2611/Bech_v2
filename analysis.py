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
from collections import Counter

MODEL_FILES = {
    'ESM1b': 'results/esm1b_results.tsv',
    'ESM2': 'results/esm2_results.tsv',
    'ESM3': 'results/esm3_results.tsv',
    'ProtT5': 'results/prott5_results.tsv',
    'ESM1b-MLM': 'results/esm1b_mlm_results.tsv',
    'ESM2-MLM': 'results/esm2_mlm_results.tsv',
    'ESM-1v-MLM': 'results/esm1v_t33_650M_UR90S_1_mlm_results.tsv',
    'ESM3-MLM': 'results/esm3_mlm_results.tsv',
    'ProtBERT-MLM': 'results/protbert_mlm_results.tsv',
    'ProtT5-MLM': 'results/prott5_mlm_results.tsv',
    'Ankh-MLM': 'results/ankh_mlm_results.tsv',
    'DNABERT-1-MLM': 'results/dnabert1_mlm_results.tsv',
    'DNABERT-2-MLM': 'results/dnabert2_mlm_results.tsv',
    'Gena-LM-MLM': 'results/genalm_mlm_results.tsv',
    'NT-v2-MLM': 'results/ntv2_mlm_results.tsv',
    'DNABERT-2': 'results/dnabert2_results.tsv',
    'HyenaDNA': 'results/hyenadna_results.tsv',
    'NT-v2': 'results/ntv2_results.tsv',
    'ProtBERT': 'results/protbert_results.tsv',
    'ESM-1v': 'results/esm1v_results.tsv',
    'Ankh': 'results/ankh_results.tsv',
    'DNABERT-1': 'results/dnabert1_results.tsv',
    'Gena-LM': 'results/genalm_results.tsv',
}

# Caduceus removed — mamba_ssm compilation failed

MODEL_INFO = {
    'ESM1b': {
        'model_id': 'esm1b_t33_650M_UR50S',
        'type': 'Protein',
        'architecture': 'Transformer Encoder (33 layers)',
        'parameters': '650M',
        'embed_layer': 'Layer 15 (of 33)',
        'embedding_dim': 1280,
        'max_input_length': '1024 tokens (~1024 aa)',
        'tokenization': 'Amino acid (1 token = 1 residue)',
        'method': 'Cosine distance',
    },
    'ESM2': {
        'model_id': 'esm2_t36_3B_UR50S',
        'type': 'Protein',
        'architecture': 'Transformer Encoder (36 layers)',
        'parameters': '3B',
        'embed_layer': 'Layer 15 (of 36)',
        'embedding_dim': 2560,
        'max_input_length': '1024 tokens (~1024 aa)',
        'tokenization': 'Amino acid (1 token = 1 residue)',
        'method': 'Cosine distance',
    },
    'ESM3': {
        'model_id': 'esm3-open',
        'type': 'Protein',
        'architecture': 'Transformer Encoder (multimodal)',
        'parameters': '1.4B',
        'embed_layer': 'Final embeddings (all tokens)',
        'embedding_dim': 1536,
        'max_input_length': '2048 tokens',
        'tokenization': 'Amino acid (1 token = 1 residue)',
        'method': 'Cosine distance',
    },
    'ProtT5': {
        'model_id': 'prot_t5_xl_half_uniref50-enc',
        'type': 'Protein',
        'architecture': 'T5 Encoder (24 layers)',
        'parameters': '3B',
        'embed_layer': 'Last hidden state',
        'embedding_dim': 1024,
        'max_input_length': '1024 tokens (~1024 aa)',
        'tokenization': 'Spaced AA ("M K T A Y...")',
        'method': 'Cosine distance',
    },
    'ESM1b-MLM': {
        'model_id': 'esm1b_t33_650M_UR50S',
        'type': 'Protein',
        'architecture': 'Transformer Encoder (33 layers)',
        'parameters': '650M',
        'embed_layer': 'MLM logits (softmax layer)',
        'embedding_dim': '33 (vocab logits)',
        'max_input_length': '1024 tokens (~1024 aa)',
        'tokenization': 'Amino acid (1 token = 1 residue)',
        'method': 'Masked marginal log-likelihood',
    },
    'DNABERT-2': {
        'model_id': 'DNABERT-2-117M',
        'type': 'Nucleotide',
        'architecture': 'BERT Encoder',
        'parameters': '117M',
        'embed_layer': 'Last hidden state (all tokens)',
        'embedding_dim': 768,
        'max_input_length': '512 BPE tokens (~2000 bp)',
        'tokenization': 'BPE (variable-length subwords)',
        'method': 'Cosine distance',
    },
    'HyenaDNA': {
        'model_id': 'hyenadna-large-1m-seqlen-hf',
        'type': 'Nucleotide',
        'architecture': 'Hyena Hierarchy (long-range)',
        'parameters': '~1B',
        'embed_layer': 'Last hidden state (all tokens)',
        'embedding_dim': 1024,
        'max_input_length': '1M bp (single-nucleotide)',
        'tokenization': 'Single nucleotide (1 token = 1 bp)',
        'method': 'Cosine distance',
    },
    'NT-v2': {
        'model_id': 'nucleotide-transformer-v2-500m-multi-species',
        'type': 'Nucleotide',
        'architecture': 'Transformer Encoder (ESM-based)',
        'parameters': '500M',
        'embed_layer': 'Last hidden state (all tokens)',
        'embedding_dim': 1280,
        'max_input_length': '2048 tokens (~12,288 bp)',
        'tokenization': '6-mer (1 token = 6 bp)',
        'method': 'Cosine distance',
    },
    'ProtBERT': {
        'model_id': 'prot_bert_bfd',
        'type': 'Protein',
        'architecture': 'BERT Encoder (30 layers)',
        'parameters': '345M',
        'embed_layer': 'Last hidden state',
        'embedding_dim': 1024,
        'max_input_length': '2048 tokens (~2048 aa)',
        'tokenization': 'Character-level AA (space-separated)',
        'method': 'Cosine distance',
    },
    'ESM-1v': {
        'model_id': 'esm1v_t33_650M_UR90S_1',
        'type': 'Protein',
        'architecture': 'Transformer Encoder (33 layers)',
        'parameters': '650M',
        'embed_layer': 'Layer 33 (of 33)',
        'embedding_dim': 1280,
        'max_input_length': '1024 tokens (~1022 aa)',
        'tokenization': 'Amino acid (1 token = 1 residue)',
        'method': 'Cosine distance',
    },
    'Ankh': {
        'model_id': 'ankh-large',
        'type': 'Protein',
        'architecture': 'T5 Encoder-Decoder (48 layers)',
        'parameters': '1.5B',
        'embed_layer': 'Encoder last hidden state',
        'embedding_dim': 1536,
        'max_input_length': '512 tokens',
        'tokenization': 'SentencePiece (vocab=144)',
        'method': 'Cosine distance',
    },
    'DNABERT-1': {
        'model_id': 'DNA_bert_6',
        'type': 'Nucleotide',
        'architecture': 'BERT Encoder (12 layers)',
        'parameters': '110M',
        'embed_layer': 'Last hidden state (all tokens)',
        'embedding_dim': 768,
        'max_input_length': '512 tokens (~3072 bp)',
        'tokenization': '6-mer k-mer (1 token = 6 bp)',
        'method': 'Cosine distance',
    },
    'Gena-LM': {
        'model_id': 'gena-lm-bert-base',
        'type': 'Nucleotide',
        'architecture': 'BERT Encoder (Pre-LN, 12 layers)',
        'parameters': '110M',
        'embed_layer': 'Last hidden state (all tokens)',
        'embedding_dim': 768,
        'max_input_length': '512 tokens (~4500 bp)',
        'tokenization': 'BPE (vocab=32k)',
        'method': 'Cosine distance',
    },
    'ESM2-MLM': {
        'model_id': 'esm2_t36_3B_UR50S',
        'type': 'Protein',
        'architecture': 'Transformer Encoder (36 layers)',
        'parameters': '3B',
        'embed_layer': 'MLM logits (softmax layer)',
        'embedding_dim': '33 (vocab logits)',
        'max_input_length': '1024 tokens (~1024 aa)',
        'tokenization': 'Amino acid (1 token = 1 residue)',
        'method': 'Masked marginal log-likelihood',
    },
    'ESM-1v-MLM': {
        'model_id': 'esm1v_t33_650M_UR90S_1',
        'type': 'Protein',
        'architecture': 'Transformer Encoder (33 layers)',
        'parameters': '650M',
        'embed_layer': 'MLM logits (softmax layer)',
        'embedding_dim': '33 (vocab logits)',
        'max_input_length': '1024 tokens (~1022 aa)',
        'tokenization': 'Amino acid (1 token = 1 residue)',
        'method': 'Masked marginal log-likelihood',
    },
    'ESM3-MLM': {
        'model_id': 'esm3-open',
        'type': 'Protein',
        'architecture': 'Transformer Encoder (multimodal)',
        'parameters': '1.4B',
        'embed_layer': 'MLM logits (sequence track)',
        'embedding_dim': '64 (seq logits)',
        'max_input_length': '2048 tokens',
        'tokenization': 'Amino acid (1 token = 1 residue)',
        'method': 'Masked marginal log-likelihood',
    },
    'ProtBERT-MLM': {
        'model_id': 'prot_bert_bfd',
        'type': 'Protein',
        'architecture': 'BERT Encoder (30 layers)',
        'parameters': '345M',
        'embed_layer': 'MLM logits (softmax layer)',
        'embedding_dim': '30 (vocab logits)',
        'max_input_length': '2048 tokens (~2048 aa)',
        'tokenization': 'Character-level AA (space-separated)',
        'method': 'Masked marginal log-likelihood',
    },
    'ProtT5-MLM': {
        'model_id': 'prot_t5_xl_uniref50',
        'type': 'Protein',
        'architecture': 'T5 Encoder-Decoder (24 layers)',
        'parameters': '3B',
        'embed_layer': 'Decoder output logits',
        'embedding_dim': '32 (vocab logits)',
        'max_input_length': '1024 tokens (~1024 aa)',
        'tokenization': 'Spaced AA ("M K T A Y...")',
        'method': 'Masked marginal log-likelihood',
    },
    'Ankh-MLM': {
        'model_id': 'ankh-large',
        'type': 'Protein',
        'architecture': 'T5 Encoder-Decoder (48 layers)',
        'parameters': '1.5B',
        'embed_layer': 'Decoder output logits',
        'embedding_dim': '144 (vocab logits)',
        'max_input_length': '512 tokens',
        'tokenization': 'SentencePiece (vocab=144)',
        'method': 'Masked marginal log-likelihood',
    },
    'DNABERT-1-MLM': {
        'model_id': 'DNA_bert_6',
        'type': 'Nucleotide',
        'architecture': 'BERT Encoder (12 layers)',
        'parameters': '110M',
        'embed_layer': 'MLM logits (softmax layer)',
        'embedding_dim': 'vocab logits',
        'max_input_length': '512 tokens (~3072 bp)',
        'tokenization': '6-mer k-mer (1 token = 6 bp)',
        'method': 'Masked marginal log-likelihood',
    },
    'DNABERT-2-MLM': {
        'model_id': 'DNABERT-2-117M',
        'type': 'Nucleotide',
        'architecture': 'BERT Encoder',
        'parameters': '117M',
        'embed_layer': 'MLM logits (softmax layer)',
        'embedding_dim': 'vocab logits',
        'max_input_length': '512 BPE tokens (~2000 bp)',
        'tokenization': 'BPE (variable-length subwords)',
        'method': 'Masked marginal log-likelihood',
    },
    'Gena-LM-MLM': {
        'model_id': 'gena-lm-bert-base',
        'type': 'Nucleotide',
        'architecture': 'BERT Encoder (Pre-LN, 12 layers)',
        'parameters': '110M',
        'embed_layer': 'MLM logits (softmax layer)',
        'embedding_dim': 'vocab logits',
        'max_input_length': '512 tokens (~4500 bp)',
        'tokenization': 'BPE (vocab=32k)',
        'method': 'Masked marginal log-likelihood',
    },
    'NT-v2-MLM': {
        'model_id': 'nucleotide-transformer-v2-500m-multi-species',
        'type': 'Nucleotide',
        'architecture': 'Transformer Encoder (ESM-based)',
        'parameters': '500M',
        'embed_layer': 'MLM logits (softmax layer)',
        'embedding_dim': 'vocab logits',
        'max_input_length': '2048 tokens (~12,288 bp)',
        'tokenization': '6-mer (1 token = 6 bp)',
        'method': 'Masked marginal log-likelihood',
    },
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
    'ProtBERT': '#17becf',
    'ESM-1v': '#bcbd22',
    'Ankh': '#e74c3c',
    'DNABERT-1': '#3498db',
    'Gena-LM': '#2ecc71',
    'ESM2-MLM': '#ff7f0e',
    'ESM-1v-MLM': '#bcbd22',
    'ESM3-MLM': '#98df8a',
    'ProtBERT-MLM': '#17becf',
    'ProtT5-MLM': '#ff9896',
    'Ankh-MLM': '#c49c94',
    'DNABERT-1-MLM': '#aec7e8',
    'DNABERT-2-MLM': '#f7b6d2',
    'Gena-LM-MLM': '#98df8a',
    'NT-v2-MLM': '#c5b0d5',
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


def plot_cosine_vs_mlm_comparison(all_metrics, output_dir):
    cosine_models = {k: v for k, v in all_metrics.items() if not k.endswith('-MLM')}
    mlm_models = {k: v for k, v in all_metrics.items() if k.endswith('-MLM')}

    base_names = []
    for k in cosine_models:
        base = k
        mlm_key = f'{base}-MLM'
        if mlm_key in mlm_models:
            base_names.append(base)

    if not base_names:
        print("No cosine+MLM pairs found for comparison plot")
        return

    x = np.arange(len(base_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    cosine_aurocs = [cosine_models[b]['AUROC'] for b in base_names]
    mlm_aurocs = [mlm_models[f'{b}-MLM']['AUROC'] for b in base_names]

    bars1 = ax.bar(x - width/2, cosine_aurocs, width, label='Cosine Distance',
                   color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, mlm_aurocs, width, label='Masked Marginal Log-Lik',
                   color='#e74c3c', edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars1, cosine_aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, mlm_aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Cosine Distance vs Masked Marginal Log-Likelihood', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(base_names, rotation=30, ha='right', fontsize=10)
    ax.set_ylim(min(min(cosine_aurocs), min(mlm_aurocs)) - 0.05, 1.01)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_vs_mlm_auroc.png'), dpi=300)
    plt.close()


def plot_roc_cosine_vs_mlm(all_dfs, output_dir):
    cosine_dfs = {k: v for k, v in all_dfs.items() if not k.endswith('-MLM')}
    mlm_dfs = {k: v for k, v in all_dfs.items() if k.endswith('-MLM')}

    pairs = []
    for k in cosine_dfs:
        mlm_key = f'{k}-MLM'
        if mlm_key in mlm_dfs:
            pairs.append((k, mlm_key))

    if not pairs:
        print("No cosine+MLM pairs for ROC comparison")
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (cos_name, mlm_name) in zip(axes, pairs):
        for name, ls in [(cos_name, '-'), (mlm_name, '--')]:
            df = all_dfs[name]
            y_true = df['label_binary'].values
            y_score = df['cosine_distance'].values
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auroc = roc_auc_score(y_true, y_score)
            method = 'MLM' if name.endswith('-MLM') else 'Cosine'
            ax.plot(fpr, tpr, linestyle=ls, color=COLORS.get(name, '#333'),
                    label=f'{method} (AUROC={auroc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR', fontsize=10)
        ax.set_ylabel('TPR', fontsize=10)
        ax.set_title(cos_name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

    plt.suptitle('ROC: Cosine Distance vs MLM', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_cosine_vs_mlm.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_info(all_metrics, output_dir):
    rows = []
    for model_name in all_metrics:
        if model_name in MODEL_INFO:
            info = MODEL_INFO[model_name]
            rows.append({
                'Model': model_name,
                'Type': info['type'],
                'Architecture': info['architecture'],
                'Parameters': info['parameters'],
                'Embedding Layer': info['embed_layer'],
                'Embed Dim': info['embedding_dim'],
                'Max Input': info['max_input_length'],
                'Tokenization': info['tokenization'],
                'Method': info['method'],
                'AUROC': all_metrics[model_name]['AUROC'],
            })

    if not rows:
        return

    info_df = pd.DataFrame(rows)
    info_df.to_csv(os.path.join(output_dir, 'model_info.csv'), index=False)
    print(f"Model info: {os.path.join(output_dir, 'model_info.csv')}")

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis('off')

    table_data = info_df[['Model', 'Type', 'Parameters', 'Embedding Layer',
                          'Max Input', 'Tokenization', 'Method', 'AUROC']].copy()
    table_data['AUROC'] = table_data['AUROC'].apply(lambda x: f'{x:.3f}')

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    header_color = '#2c3e50'
    header_text_color = 'white'
    for j in range(len(table_data.columns)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text_color, fontweight='bold')

    for i in range(len(table_data)):
        row_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        for j in range(len(table_data.columns)):
            cell = table[i + 1, j]
            cell.set_facecolor(row_color)
            if j == 0:
                cell.set_text_props(fontweight='bold')

    auroc_col = list(table_data.columns).index('AUROC')
    aurocs = info_df['AUROC'].values
    best_idx = np.argmax(aurocs)
    table[best_idx + 1, auroc_col].set_facecolor('#d4edda')
    table[best_idx + 1, auroc_col].set_text_props(fontweight='bold', color='#155724')

    ax.set_title('Model Architecture, Parameters, and Embedding Details',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_info.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_data_analysis(output_dir):
    df = pd.read_csv('missense_variants.tsv', sep='\t')
    print(f"\n=== Dataset Analysis ===")
    print(f"Total variants: {len(df)}")
    print(f"Unique proteins: {df['refseq_accession'].nunique()}")
    print(f"Unique genes: {df['gene_symbol'].nunique()}")

    aa_order = list('ACDEFGHIKLMNPQRSTVWY')
    aa_names = {
        'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe',
        'G': 'Gly', 'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu',
        'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln', 'R': 'Arg',
        'S': 'Ser', 'T': 'Thr', 'V': 'Val', 'W': 'Trp', 'Y': 'Tyr',
    }

    fig = plt.figure(figsize=(20, 16))

    ax1 = fig.add_subplot(2, 3, 1)
    counts = df['clinical_significance'].value_counts()
    colors_pie = ['#dc3545', '#28a745']
    wedges, texts, autotexts = ax1.pie(
        counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=colors_pie, startangle=90, textprops={'fontsize': 11}
    )
    for t in autotexts:
        t.set_fontweight('bold')
    ax1.set_title(f'Class Distribution\n(n={len(df):,})', fontsize=13, fontweight='bold')

    ax2 = fig.add_subplot(2, 3, 2)
    gene_counts = df.groupby('gene_symbol').size().sort_values(ascending=False)
    top_n = 20
    top_genes = gene_counts.head(top_n)
    bars = ax2.barh(range(top_n), top_genes.values[::-1], color='#3498db', edgecolor='white')
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_genes.index[::-1], fontsize=9)
    ax2.set_xlabel('Number of Variants', fontsize=11)
    ax2.set_title(f'Top {top_n} Genes by Variant Count', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, top_genes.values[::-1]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 str(val), va='center', fontsize=8)
    sns.despine(ax=ax2)

    ax3 = fig.add_subplot(2, 3, 3)
    gene_counts_per = df.groupby('refseq_accession').size()
    ax3.hist(gene_counts_per.values, bins=50, color='#9b59b6', edgecolor='white', alpha=0.8)
    ax3.set_xlabel('Variants per Protein', fontsize=11)
    ax3.set_ylabel('Number of Proteins', fontsize=11)
    ax3.set_title(f'Variant Count per Protein\n(median={np.median(gene_counts_per.values):.0f}, '
                  f'max={gene_counts_per.max()})', fontsize=13, fontweight='bold')
    ax3.axvline(x=np.median(gene_counts_per.values), color='red', linestyle='--', alpha=0.7, label='Median')
    ax3.legend()
    sns.despine(ax=ax3)

    ax4 = fig.add_subplot(2, 3, 4)
    sub_matrix = pd.DataFrame(0, index=aa_order, columns=aa_order)
    for _, row in df.iterrows():
        w, m = row['wt_aa'], row['mut_aa']
        if w in aa_order and m in aa_order:
            sub_matrix.loc[w, m] += 1
    np.fill_diagonal(sub_matrix.values, 0)
    sns.heatmap(sub_matrix, cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Count'},
                xticklabels=[aa_names.get(a, a) for a in aa_order],
                yticklabels=[aa_names.get(a, a) for a in aa_order],
                linewidths=0.5, linecolor='white')
    ax4.set_xlabel('Mutant', fontsize=11)
    ax4.set_ylabel('Wild-type', fontsize=11)
    ax4.set_title('Amino Acid Substitution Matrix', fontsize=13, fontweight='bold')
    plt.sca(ax4)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    ax5 = fig.add_subplot(2, 3, 5)
    hydrophobic = set('AILMFWV')
    positive = set('RHK')
    negative = set('DE')
    polar = set('STNQ')
    special = set('CGP')
    aromatic = set('FWY')

    def classify_change(w, m):
        if w in hydrophobic and m in hydrophobic:
            return 'Hydrophobic'
        elif w in positive and m in positive:
            return 'Positive'
        elif w in negative and m in negative:
            return 'Negative'
        elif w in polar and m in polar:
            return 'Polar'
        elif w in aromatic and m in aromatic:
            return 'Aromatic'
        else:
            return 'Cross-property'

    df['change_type'] = df.apply(lambda r: classify_change(r['wt_aa'], r['mut_aa']), axis=1)
    change_counts = df.groupby(['change_type', 'clinical_significance']).size().unstack(fill_value=0)
    change_order = ['Hydrophobic', 'Polar', 'Positive', 'Negative', 'Aromatic', 'Cross-property']
    change_counts = change_counts.reindex([c for c in change_order if c in change_counts.index])
    change_counts.plot(kind='bar', ax=ax5, color=['#dc3545', '#28a745'], edgecolor='white')
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Mutation Type by Significance', fontsize=13, fontweight='bold')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    ax5.legend(fontsize=9)
    sns.despine(ax=ax5)

    ax6 = fig.add_subplot(2, 3, 6)
    wt_counts = df['wt_aa'].value_counts().reindex(aa_order).fillna(0)
    mut_counts = df['mut_aa'].value_counts().reindex(aa_order).fillna(0)
    x = np.arange(len(aa_order))
    width = 0.35
    ax6.bar(x - width/2, wt_counts.values, width, label='Wild-type', color='#3498db', alpha=0.8)
    ax6.bar(x + width/2, mut_counts.values, width, label='Mutant', color='#e74c3c', alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels([aa_names.get(a, a) for a in aa_order], rotation=45, ha='right', fontsize=8)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('Amino Acid Frequency\n(WT vs Mutant)', fontsize=13, fontweight='bold')
    ax6.legend()
    sns.despine(ax=ax6)

    plt.suptitle('ClinVar Missense Variant Dataset Analysis', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Pathogenic: {(df['clinical_significance']=='Pathogenic').sum()}")
    print(f"  Benign: {(df['clinical_significance']=='Benign').sum()}")
    print(f"  Top 10 genes:")
    for g, c in gene_counts.head(10).items():
        print(f"    {g}: {c}")
    print(f"  Dataset analysis: {os.path.join(output_dir, 'dataset_analysis.png')}")

    df.drop(columns=['change_type'], inplace=True)


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

    cosine_only_dfs = {k: v for k, v in all_dfs.items() if not k.endswith('-MLM')}
    cosine_only_metrics = {k: v for k, v in all_metrics.items() if not k.endswith('-MLM')}
    mlm_only_dfs = {k: v for k, v in all_dfs.items() if k.endswith('-MLM')}
    mlm_only_metrics = {k: v for k, v in all_metrics.items() if k.endswith('-MLM')}

    summary_df = pd.DataFrame(all_metrics).T
    summary_df.to_csv(os.path.join(output_dir, 'benchmark_summary.csv'))
    print(f"\nSummary: {os.path.join(output_dir, 'benchmark_summary.csv')}")

    print("\n--- Cosine Distance Plots ---")
    plot_auroc_comparison(cosine_only_metrics, output_dir)
    plot_distance_distributions(cosine_only_dfs, output_dir)
    plot_roc_curves(cosine_only_dfs, output_dir)
    plot_pr_curves(cosine_only_dfs, output_dir)
    plot_boxplot(cosine_only_dfs, output_dir)
    per_gene_analysis(cosine_only_dfs, output_dir)
    plot_model_info(cosine_only_metrics, output_dir)

    print("\n--- Combined Cosine vs MLM Plots ---")
    plot_cosine_vs_mlm_comparison(all_metrics, output_dir)
    plot_roc_cosine_vs_mlm(all_dfs, output_dir)

    plot_data_analysis(output_dir)

    print(f"\nAll plots saved to {output_dir}/")
    print("Files:")
    for f in os.listdir(output_dir):
        if f.endswith('.png') or f.endswith('.csv'):
            print(f"  {f}")


if __name__ == '__main__':
    main()
