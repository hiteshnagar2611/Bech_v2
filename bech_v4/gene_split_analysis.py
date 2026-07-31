"""
Compare Old (Random) vs New (Gene-Disjoint) Split Results
Generates paper figures showing the effect of proper gene-disjoint evaluation.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BECH_DIR = BASE_DIR


KEY_MAP = {'ProtBERT': 'ProtBERT-Cos', 'NT-v2': 'NT-v2-Cos'}


def load_results():
    old_d1 = json.load(open(os.path.join(BECH_DIR, 'dataset1_clinvar_only', 'meta_model', 'results.json')))
    old_d2 = json.load(open(os.path.join(BECH_DIR, 'dataset2_jan2025', 'meta_model', 'results.json')))
    new_d1 = json.load(open(os.path.join(BECH_DIR, 'dataset1_clinvar_only', 'meta_model_gene_split', 'results.json')))

    old_ind = old_d1['individual_aurocs']
    remapped = {}
    for k, v in old_ind.items():
        remapped[KEY_MAP.get(k, k)] = v
    old_d1['individual_aurocs'] = remapped

    return old_d1, old_d2, new_d1


def make_comparison_table(old_d1, old_d2, new_d1):
    print("\n" + "="*80)
    print("COMPARISON: Old (Random Split) vs New (Gene-Disjoint Split)")
    print("="*80)

    print(f"\n{'Model':<25} {'Old D1':>10} {'New D1':>10} {'Delta':>10} {'D2':>10}")
    print("-"*65)

    old_ind = old_d1['individual_aurocs']
    new_ind = new_d1['individual_aurocs']
    d2_ind = new_d1.get('d2_individual_aurocs', old_d2.get('individual_aurocs', {}))

    models = list(old_ind.keys())
    for m in models:
        old_val = old_ind.get(m, 0)
        new_val = new_ind.get(m, 0)
        delta = new_val - old_val
        d2_val = d2_ind.get(m, 0)
        print(f"  {m:<23} {old_val:>10.4f} {new_val:>10.4f} {delta:>+10.4f} {d2_val:>10.4f}")

    print("\nKey findings:")
    print(f"  Gene-disjoint AUROC drop (Meta-Learner): {old_ind.get('Meta-Learner (Ours)', 0) - new_ind.get('Meta-Learner (Ours)', 0):.4f}")
    print(f"  Meta-Learner still beats best individual on gene-disjoint: {new_ind.get('Meta-Learner (Ours)', 0) > max(v for k,v in new_ind.items() if k != 'Meta-Learner (Ours)')}")


def plot_comparison(old_d1, new_d1, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    old_ind = old_d1['individual_aurocs']
    new_ind = new_d1['individual_aurocs']

    models = [k for k in old_ind.keys() if k != 'Meta-Learner (Ours)']
    models.append('Meta-Learner (Ours)')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(models))
    width = 0.35

    old_vals = [old_ind.get(m, 0) for m in models]
    new_vals = [new_ind.get(m, 0) for m in models]

    bars1 = axes[0].bar(x - width/2, old_vals, width, label='Random Split (old)', color='#2196F3', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, new_vals, width, label='Gene-Disjoint (new)', color='#FF5722', alpha=0.8)

    axes[0].set_ylabel('AUROC', fontsize=12)
    axes[0].set_title('Old vs New Split: Individual Models', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace(' (Ours)', '\n(Ours)') for m in models], rotation=45, ha='right', fontsize=8)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    deltas = [new_val - old_val for old_val, new_val in zip(old_vals, new_vals)]
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]
    axes[1].bar(x, deltas, color=colors, alpha=0.8)
    axes[1].set_ylabel('AUROC Change (New - Old)', fontsize=12)
    axes[1].set_title('Performance Change: Gene-Disjoint vs Random', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace(' (Ours)', '\n(Ours)') for m in models], rotation=45, ha='right', fontsize=8)
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].grid(axis='y', alpha=0.3)

    for i, d in enumerate(deltas):
        axes[1].annotate(f'{d:+.4f}', xy=(x[i], d), xytext=(0, 8 if d >= 0 else -15),
                        textcoords='offset points', ha='center', fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_gene_split_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_dir}/fig_gene_split_comparison.png")

    # Summary table figure
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = [['Model', 'Random Split', 'Gene-Disjoint', 'Delta']]
    for m in models:
        ov = old_ind.get(m, 0)
        nv = new_ind.get(m, 0)
        delta = nv - ov
        table_data.append([m, f'{ov:.4f}', f'{nv:.4f}', f'{delta:+.4f}'])

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2196F3')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 3:
            val = float(cell.get_text().get_text().replace('+', ''))
            if val < -0.01:
                cell.set_facecolor('#FFCDD2')
            elif val > 0.01:
                cell.set_facecolor('#C8E6C9')
            else:
                cell.set_facecolor('#FFF9C4')

    ax.set_title('Random Split vs Gene-Disjoint Split: AUROC Comparison', fontsize=13, fontweight='bold', pad=20)
    fig2.savefig(os.path.join(out_dir, 'fig_gene_split_table.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_dir}/fig_gene_split_table.png")


def main():
    old_d1, old_d2, new_d1 = load_results()
    make_comparison_table(old_d1, old_d2, new_d1)
    out_dir = os.path.join(BASE_DIR, 'gene_split_results')
    plot_comparison(old_d1, new_d1, out_dir)


if __name__ == '__main__':
    main()
