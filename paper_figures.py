import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = 'figures/paper'
os.makedirs(OUT, exist_ok=True)

summary = pd.read_csv('figures/benchmark_summary.csv', index_col=0)

COLORS_COSINE = {
    'ESM1b': '#4C72B0', 'ESM2': '#55A868', 'ESM3': '#C44E52',
    'ProtT5': '#8172B2', 'ProtBERT': '#CCB974', 'ESM-1v': '#64B5CD',
    'Ankh': '#DA8BC3', 'DNABERT-1': '#8C8C8C', 'DNABERT-2': '#B25124',
    'Gena-LM': '#89C56A', 'NT-v2': '#5B9BD5', 'HyenaDNA': '#ED7D31',
}
COLORS_MLM = {
    'ESM1b-MLM': '#4C72B0', 'ESM2-MLM': '#55A868', 'ESM3-MLM': '#C44E52',
    'ProtBERT-MLM': '#CCB974', 'ESM-1v-MLM': '#64B5CD',
    'ProtT5-MLM': '#8172B2', 'Ankh-MLM': '#DA8BC3',
    'DNABERT-1-MLM': '#8C8C8C', 'DNABERT-2-MLM': '#B25124',
    'Gena-LM-MLM': '#89C56A', 'NT-v2-MLM': '#5B9BD5',
}

def load_tsv(path):
    df = pd.read_csv(path, sep='\t')
    df['label_binary'] = (df['label'] == 'Pathogenic').astype(int)
    return df


def make_pairs():
    pairs = []
    for base in ['ESM1b', 'ESM2', 'ESM3', 'ProtBERT', 'ESM-1v',
                  'ProtT5', 'Ankh', 'DNABERT-1', 'DNABERT-2', 'Gena-LM', 'NT-v2']:
        mlm = f'{base}-MLM'
        cos_path = f'results/{base.lower().replace("-", "")}_results.tsv'
        mlm_path = f'results/{base.lower().replace("-", "")}_mlm_results.tsv'
        if base == 'ESM-1v':
            cos_path = 'results/esm1v_results.tsv'
            mlm_path = 'results/esm1v_t33_650M_UR90S_1_mlm_results.tsv'
        if base == 'ESM1b':
            cos_path = 'results/esm1b_results.tsv'
            mlm_path = 'results/esm1b_mlm_results.tsv'
        if base == 'ESM2':
            cos_path = 'results/esm2_results.tsv'
            mlm_path = 'results/esm2_mlm_results.tsv'
        if base == 'ESM3':
            cos_path = 'results/esm3_results.tsv'
            mlm_path = 'results/esm3_mlm_results.tsv'
        if base == 'ProtBERT':
            cos_path = 'results/protbert_results.tsv'
            mlm_path = 'results/protbert_mlm_results.tsv'
        if base == 'ProtT5':
            cos_path = 'results/prott5_results.tsv'
            mlm_path = 'results/prott5_mlm_results.tsv'
        if base == 'Ankh':
            cos_path = 'results/ankh_results.tsv'
            mlm_path = 'results/ankh_mlm_results.tsv'
        if base == 'DNABERT-1':
            cos_path = 'results/dnabert1_results.tsv'
            mlm_path = 'results/dnabert1_mlm_results.tsv'
        if base == 'DNABERT-2':
            cos_path = 'results/dnabert2_results.tsv'
            mlm_path = 'results/dnabert2_mlm_results.tsv'
        if base == 'Gena-LM':
            cos_path = 'results/genalm_results.tsv'
            mlm_path = 'results/genalm_mlm_results.tsv'
        if base == 'NT-v2':
            cos_path = 'results/ntv2_results.tsv'
            mlm_path = 'results/ntv2_mlm_results.tsv'
        if os.path.exists(cos_path) and os.path.exists(mlm_path):
            pairs.append((base, cos_path, mlm_path))
    return pairs


def fig1_auroc_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.2, 1]})

    ax = axes[0]
    pro_models = ['ESM1b', 'ESM2', 'ESM3', 'ProtT5', 'ProtBERT', 'ESM-1v', 'Ankh']
    pro_mlm = [f'{m}-MLM' for m in pro_models]

    cos_aurocs = [summary.loc[m, 'AUROC'] if m in summary.index else 0 for m in pro_models]
    mlm_aurocs = [summary.loc[m, 'AUROC'] if m in summary.index else 0 for m in pro_mlm]

    x = np.arange(len(pro_models))
    w = 0.35
    bars1 = ax.bar(x - w/2, cos_aurocs, w, label='Cosine Distance', color='#4C72B0', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, mlm_aurocs, w, label='MLM (P$_{wt}$ - P$_{mut}$)', color='#C44E52', edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars1, cos_aurocs):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#4C72B0')
    for bar, val in zip(bars2, mlm_aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#C44E52')

    ax.set_xticks(x)
    ax.set_xticklabels(pro_models, rotation=30, ha='right')
    ax.set_ylabel('AUROC')
    ax.set_title('A. Protein Language Models', fontweight='bold')
    ax.set_ylim(0.4, 1.02)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.legend(loc='lower right', framealpha=0.9)

    ax = axes[1]
    dna_models = ['DNABERT-1', 'DNABERT-2', 'Gena-LM', 'NT-v2', 'HyenaDNA']
    dna_mlm = [f'{m}-MLM' for m in dna_models if f'{m}-MLM' in summary.index]

    dna_cos = [summary.loc[m, 'AUROC'] if m in summary.index else 0 for m in dna_models]
    dna_mlm_a = [summary.loc[m, 'AUROC'] if m in summary.index else 0 for m in dna_mlm]
    dna_labels = dna_models[:len(dna_mlm)]

    x = np.arange(len(dna_labels))
    bars1 = ax.bar(x - w/2, dna_cos[:len(dna_labels)], w, label='Cosine Distance', color='#4C72B0', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, dna_mlm_a, w, label='MLM', color='#C44E52', edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars1, dna_cos[:len(dna_labels)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#4C72B0')
    for bar, val in zip(bars2, dna_mlm_a):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#C44E52')

    ax.set_xticks(x)
    ax.set_xticklabels(dna_labels, rotation=30, ha='right')
    ax.set_ylabel('AUROC')
    ax.set_title('B. Nucleotide Language Models', fontweight='bold')
    ax.set_ylim(0.4, 1.02)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.legend(loc='lower right', framealpha=0.9)

    plt.suptitle('Zero-Shot Pathogenicity Prediction on ClinVar\nCosine Distance vs Masked Marginal Log-Likelihood',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig1_auroc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'fig1_auroc_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT}/fig1_auroc_comparison.png")


def fig2_roc_top_models():
    pairs = make_pairs()
    top_pairs = []
    for base, cos_path, mlm_path in pairs:
        cos_df = load_tsv(cos_path)
        mlm_df = load_tsv(mlm_path)
        cos_auroc = roc_auc_score(cos_df['label_binary'], cos_df['cosine_distance'])
        mlm_auroc = roc_auc_score(mlm_df['label_binary'], mlm_df['cosine_distance'])
        if mlm_auroc > 0.7:
            top_pairs.append((base, cos_df, mlm_df, cos_auroc, mlm_auroc))

    top_pairs.sort(key=lambda x: -x[4])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for base, cos_df, mlm_df, cos_a, mlm_a in top_pairs:
        fpr, tpr, _ = roc_curve(mlm_df['label_binary'], mlm_df['cosine_distance'])
        ax.plot(fpr, tpr, label=f'{base} (AUROC={mlm_a:.3f})', linewidth=1.8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('A. MLM Scoring (Top Models)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1]
    for base, cos_df, mlm_df, cos_a, mlm_a in top_pairs:
        fpr, tpr, _ = roc_curve(cos_df['label_binary'], cos_df['cosine_distance'])
        ax.plot(fpr, tpr, label=f'{base} (AUROC={cos_a:.3f})', linewidth=1.8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('B. Cosine Distance (All Models)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.suptitle('ROC Curves for Zero-Shot Pathogenicity Prediction', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig2_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'fig2_roc_curves.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT}/fig2_roc_curves.png")


def fig3_gain_from_mlm():
    pairs = make_pairs()
    data = []
    for base, cos_path, mlm_path in pairs:
        cos_df = load_tsv(cos_path)
        mlm_df = load_tsv(mlm_path)
        cos_a = roc_auc_score(cos_df['label_binary'], cos_df['cosine_distance'])
        mlm_a = roc_auc_score(mlm_df['label_binary'], mlm_df['cosine_distance'])
        data.append({'Model': base, 'Cosine': cos_a, 'MLM': mlm_a, 'Gain': mlm_a - cos_a})

    df = pd.DataFrame(data).sort_values('Gain', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#C44E52' if g > 0 else '#4C72B0' for g in df['Gain']]
    bars = ax.barh(df['Model'], df['Gain'], color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    for bar, val in zip(bars, df['Gain']):
        x_pos = bar.get_width() + 0.005 if val > 0 else bar.get_width() - 0.005
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', ha=ha, va='center', fontsize=9, fontweight='bold')

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('AUROC Gain (MLM - Cosine)')
    ax.set_title('Performance Gain from MLM Scoring', fontweight='bold')
    ax.set_xlim(-0.4, 0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig3_mlm_gain.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'fig3_mlm_gain.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT}/fig3_mlm_gain.png")


def fig4_summary_table():
    pairs = make_pairs()
    rows = []
    for base, cos_path, mlm_path in pairs:
        cos_df = load_tsv(cos_path)
        mlm_df = load_tsv(mlm_path)
        cos_a = roc_auc_score(cos_df['label_binary'], cos_df['cosine_distance'])
        mlm_a = roc_auc_score(mlm_df['label_binary'], mlm_df['cosine_distance'])

        info = summary.loc[base] if base in summary.index else {}
        rows.append({
            'Model': base,
            'Type': 'Protein' if base in ['ESM1b','ESM2','ESM3','ProtT5','ProtBERT','ESM-1v','Ankh'] else 'Nucleotide',
            'Params': info.get('Parameters', '?'),
            'Cosine AUROC': f'{cos_a:.3f}',
            'MLM AUROC': f'{mlm_a:.3f}',
            'Gain': f'{mlm_a - cos_a:+.3f}',
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for j in range(len(df.columns)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    for i in range(len(df)):
        for j in range(len(df.columns)):
            cell = table[i + 1, j]
            cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
            if j == 0:
                cell.set_text_props(fontweight='bold')

    gain_col = list(df.columns).index('Gain')
    for i in range(len(df)):
        gain_val = float(df.iloc[i]['Gain'])
        cell = table[i + 1, gain_col]
        if gain_val > 0.1:
            cell.set_facecolor('#d4edda')
            cell.set_text_props(color='#155724', fontweight='bold')
        elif gain_val < -0.01:
            cell.set_facecolor('#f8d7da')
            cell.set_text_props(color='#721c24')

    best_cos = df['Cosine AUROC'].astype(float).idxmax()
    best_mlm = df['MLM AUROC'].astype(float).idxmax()
    table[best_cos + 1, list(df.columns).index('Cosine AUROC')].set_facecolor('#cce5ff')
    table[best_mlm + 1, list(df.columns).index('MLM AUROC')].set_facecolor('#cce5ff')

    ax.set_title('Model Comparison: AUROC for Zero-Shot Pathogenicity Prediction',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig4_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'fig4_summary_table.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUT}/fig4_summary_table.png")


if __name__ == '__main__':
    print("Generating paper-ready figures...")
    fig1_auroc_comparison()
    fig2_roc_top_models()
    fig3_gain_from_mlm()
    fig4_summary_table()
    print("Done!")
