import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 20))
ax.set_xlim(0, 16)
ax.set_ylim(0, 20)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False, textcolor='white', alpha=0.9, style='round,pad=0.05'):
    box = FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=color, edgecolor='#333333',
                         linewidth=1.5, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color=textcolor, zorder=3, wrap=True,
            bbox=dict(boxstyle='round,pad=0.01', facecolor='none', edgecolor='none'))

def draw_arrow(ax, x1, y1, x2, y2, color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2, shrinkA=2, shrinkB=2))

title_y = 19.3
ax.text(8, title_y, 'ClinVar Zero-Shot Pathogenicity Benchmark Pipeline',
        ha='center', va='center', fontsize=16, fontweight='bold', color='#2c3e50')

ax.text(8, title_y - 0.45, 'Benchmarking Protein & Nucleotide Language Models on 7,991 ClinVar Missense Variants',
        ha='center', va='center', fontsize=10, color='#666666')

section_color = '#34495e'

draw_box(ax, 5.5, 18.0, 5, 0.55, 'DATA COLLECTION', section_color, fontsize=11, bold=True)

draw_box(ax, 1.5, 16.9, 4.2, 0.65, 'ClinVar Database\nclinvar_rcv_enriched.tsv.gz', '#3498db', fontsize=9)
draw_box(ax, 5.8, 16.9, 4.4, 0.65, 'Filter: Missense Variants\n7,991 (5,328 Path. + 2,663 Benign)', '#2ecc71', fontsize=9)
draw_box(ax, 10.3, 16.9, 4.2, 0.65, 'Protein Sequences (RefSeq)\n2,381 unique proteins', '#e67e22', fontsize=9)

draw_arrow(ax, 3.6, 16.9, 3.6, 17.55)
draw_arrow(ax, 8.0, 16.9, 8.0, 17.55)
draw_arrow(ax, 12.4, 16.9, 12.4, 17.55)
draw_arrow(ax, 5.9, 17.22, 5.8, 17.22)

draw_box(ax, 5.5, 16.1, 5, 0.55, 'DATA PREPARATION', section_color, fontsize=11, bold=True)

draw_box(ax, 0.8, 15.0, 3.5, 0.65, 'data_prep.py\nParse ClinVar → TSV', '#2980b9', fontsize=9)
draw_box(ax, 4.7, 15.0, 3.5, 0.65, 'missense_variants.tsv\nWT/Mut sequences', '#27ae60', fontsize=9)
draw_box(ax, 8.7, 15.0, 3.5, 0.65, 'prepare_dna_data.py\nNucleotide contexts (~6kb)', '#d35400', fontsize=9)
draw_box(ax, 12.5, 15.0, 2.8, 0.65, 'dna_variants.tsv\n7,954 variants', '#c0392b', fontsize=9)

draw_arrow(ax, 3.5, 16.1, 2.5, 15.65)
draw_arrow(ax, 5.5, 16.1, 6.4, 15.65)
draw_arrow(ax, 8.0, 16.1, 10.4, 15.65)
draw_arrow(ax, 8.2, 15.32, 8.7, 15.32)
draw_arrow(ax, 12.4, 15.32, 12.5, 15.32)

draw_box(ax, 5.5, 14.1, 5, 0.55, 'MODEL LOADING & SETUP', section_color, fontsize=11, bold=True)

draw_box(ax, 1.0, 12.9, 6.5, 0.75, 'PROTEIN MODELS\nESM1b (650M) | ESM2 (3B) | ESM3 (1.4B) | ProtT5 (3B)', '#8e44ad', fontsize=9, bold=True)
draw_box(ax, 8.5, 12.9, 6.5, 0.75, 'DNA MODELS\nDNABERT-2 (117M) | HyenaDNA (~1B) | NT-v2 (500M)', '#c0392b', fontsize=9, bold=True)

draw_arrow(ax, 3.5, 14.1, 4.2, 13.65)
draw_arrow(ax, 12.0, 14.1, 11.8, 13.65)

draw_box(ax, 5.5, 12.0, 5, 0.55, 'EMBEDDING EXTRACTION', section_color, fontsize=11, bold=True)

draw_box(ax, 0.5, 10.8, 7.2, 0.75, 'Protein: Embed WT & Mut at variant position (tok_pos = aa_pos + 1)\nLayer 15 (ESM1b/ESM2) | Final (ESM3/ProtT5)', '#2980b9', fontsize=8)
draw_box(ax, 8.3, 10.8, 7.2, 0.75, 'DNA: Embed ~6002bp nucleotide context centered on variant\nDNABERT-2 (BPE) | HyenaDNA (1bp) | NT-v2 (6-mer)', '#e74c3c', fontsize=8)

draw_arrow(ax, 3.5, 12.0, 4.1, 11.55)
draw_arrow(ax, 12.0, 12.0, 11.9, 11.55)

draw_box(ax, 5.5, 9.9, 5, 0.55, 'SCORING', section_color, fontsize=11, bold=True)

draw_box(ax, 2.0, 8.8, 5.0, 0.65, 'Cosine Distance\n1 - cos_sim(WT_embed, Mut_embed)', '#3498db', fontsize=9)
draw_box(ax, 9.0, 8.8, 5.0, 0.65, 'Masked Marginal Log-Likelihood\nP(wt) - P(mut) at variant position', '#9b59b6', fontsize=9)

draw_arrow(ax, 4.5, 9.9, 4.5, 9.45)
draw_arrow(ax, 11.5, 9.9, 11.5, 9.45)

ax.text(6.8, 8.4, 'ESM1b-MLM', ha='center', fontsize=8, color='#9b59b6', style='italic')
ax.text(4.5, 8.4, 'All others', ha='center', fontsize=8, color='#3498db', style='italic')

draw_box(ax, 5.5, 7.9, 5, 0.55, 'EVALUATION', section_color, fontsize=11, bold=True)

draw_box(ax, 1.0, 6.6, 3.5, 0.85, 'Classification\nAUROC | AUPRC | F1\nAccuracy | Precision | Recall', '#27ae60', fontsize=8)
draw_box(ax, 4.8, 6.6, 3.5, 0.85, 'Statistical\nMann-Whitney U\nSpearman Correlation', '#f39c12', fontsize=8)
draw_box(ax, 8.7, 6.6, 3.0, 0.85, 'ROC Curves\nPR Curves\nPer-Gene AUROC', '#1abc9c', fontsize=8)
draw_box(ax, 12.0, 6.6, 3.0, 0.85, 'Distributions\nBoxplots\nSubstitution Matrix', '#e67e22', fontsize=8)

draw_arrow(ax, 3.0, 7.9, 2.7, 7.45)
draw_arrow(ax, 6.5, 7.9, 6.5, 7.45)
draw_arrow(ax, 10.2, 7.9, 10.2, 7.45)
draw_arrow(ax, 13.5, 7.9, 13.5, 7.45)

draw_box(ax, 5.5, 5.7, 5, 0.55, 'PLOTTING & OUTPUT', section_color, fontsize=11, bold=True)

draw_box(ax, 1.5, 4.4, 3.0, 0.75, 'dataset_analysis.png\nClass dist | Gene counts\nAA substitution matrix', '#34495e', fontsize=8)
draw_box(ax, 4.8, 4.4, 3.2, 0.75, 'benchmark_summary.csv\nAll metrics for 8 models\nHighlighted best values', '#34495e', fontsize=8)
draw_box(ax, 8.3, 4.4, 3.2, 0.75, 'roc_curves.png | pr_curves.png\nauroc_comparison.png\ndistance_distributions.png', '#34495e', fontsize=8)
draw_box(ax, 11.8, 4.4, 3.0, 0.75, 'model_info.png\nArchitecture | Parameters\nEmbed layers | Max input', '#34495e', fontsize=8)

draw_arrow(ax, 4.5, 5.7, 3.0, 5.15)
draw_arrow(ax, 6.5, 5.7, 6.4, 5.15)
draw_arrow(ax, 10.0, 5.7, 9.9, 5.15)
draw_arrow(ax, 13.5, 5.7, 13.3, 5.15)

draw_box(ax, 5.5, 3.5, 5, 0.55, 'RESULTS', section_color, fontsize=11, bold=True)

draw_box(ax, 1.5, 2.0, 13.0, 1.1, '', '#ecf0f1', fontsize=9, alpha=0.6)
ax.add_patch(FancyBboxPatch((1.5, 2.0), 13.0, 1.1, boxstyle='round,pad=0.05',
             facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=1.5, alpha=0.6, zorder=1))

results_text = (
    "AUROC:  ESM1b-MLM: 0.889  |  ESM3: 0.759  |  ESM1b: 0.587\n"
    "        ProtT5: 0.529  |  ESM2: 0.530  |  NT-v2: 0.514\n"
    "        DNABERT-2: 0.491  |  HyenaDNA: 0.481\n\n"
    "Best: ESM1b-MLM (Masked Marginal Log-Llikelihood) — far outperforms all cosine distance methods"
)
ax.text(8, 2.55, results_text, ha='center', va='center', fontsize=9, fontfamily='monospace',
        color='#2c3e50', zorder=3)

draw_arrow(ax, 8.0, 3.5, 8.0, 3.1)

ax.text(8, 1.4, 'Infrastructure: PBS cluster (gnode6) | 8× A100-SXM4-40GB | CUDA 12.4 | Python 3.10 + PyTorch 2.4',
        ha='center', fontsize=8, color='#999999')

ax.text(8, 1.1, 'Data: ClinVar (NCBI) | 7,991 missense variants | 2,381 proteins | 2,021 genes',
        ha='center', fontsize=8, color='#999999')

plt.tight_layout()
plt.savefig('figures/project_pipeline.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved figures/project_pipeline.png")
