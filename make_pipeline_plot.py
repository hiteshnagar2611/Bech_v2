import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 22))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis('off')

def draw_box(ax, cx, cy, w, h, text, color, fontsize=9, bold=False, textcolor='white', alpha=0.95):
    x = cx - w/2
    y = cy - h/2
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', facecolor=color,
                         edgecolor='#444444', linewidth=1.5, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color=textcolor, zorder=3, linespacing=1.4)

def arrow_down(ax, cx, top_y, bot_y, color='#666666'):
    ax.annotate('', xy=(cx, bot_y), xytext=(cx, top_y),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.2, shrinkA=1, shrinkB=1))

def arrow_split(ax, cx, top_y, targets, bot_y, color='#666666'):
    for tx in targets:
        ax.annotate('', xy=(tx, bot_y), xytext=(cx, top_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8, shrinkA=2, shrinkB=2))

def arrow_merge(ax, cx, bot_y, sources, top_y, color='#666666'):
    for sx in sources:
        ax.annotate('', xy=(cx, bot_y), xytext=(sx, top_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8, shrinkA=2, shrinkB=2))

# --- TITLE ---
ax.text(7, 21.5, 'ClinVar Zero-Shot Pathogenicity Benchmark Pipeline',
        ha='center', fontsize=17, fontweight='bold', color='#2c3e50')
ax.text(7, 21.1, 'Protein & Nucleotide Language Models  |  7,991 Missense Variants  |  2,381 Proteins',
        ha='center', fontsize=10, color='#777777')

# ===== SECTION 1: DATA COLLECTION =====
sec_y = 20.3
draw_box(ax, 7, sec_y, 5, 0.5, 'DATA COLLECTION', '#2c3e50', fontsize=12, bold=True)
arrow_down(ax, 7, sec_y - 0.25, 19.7)

box_y = 19.4
draw_box(ax, 3.0, box_y, 4.5, 0.55, 'ClinVar Database\nclinvar_rcv_enriched.tsv.gz (31 MB)', '#3498db', fontsize=8)
draw_box(ax, 7.5, box_y, 4.0, 0.55, 'Filter: Missense Only\n7,991 (5,328 Path. + 2,663 Benign)', '#27ae60', fontsize=8)
draw_box(ax, 11.8, box_y, 3.0, 0.55, 'RefSeq Protein Seqs\n2,381 unique', '#e67e22', fontsize=8)

# horizontal connections between collection boxes
ax.annotate('', xy=(5.25, box_y), xytext=(5.55, box_y),
            arrowprops=dict(arrowstyle='<->', color='#999', lw=1.2))
ax.annotate('', xy=(9.5, box_y), xytext=(10.0, box_y),
            arrowprops=dict(arrowstyle='<->', color='#999', lw=1.2))

# ===== SECTION 2: DATA PREPARATION =====
sec_y = 18.5
draw_box(ax, 7, sec_y, 5, 0.5, 'DATA PREPARATION', '#2c3e50', fontsize=12, bold=True)
arrow_down(ax, 7, box_y - 0.275, sec_y + 0.25)

box_y = 17.65
draw_box(ax, 2.8, box_y, 4.0, 0.55, 'data_prep.py\nParse ClinVar → missense_variants.tsv', '#2980b9', fontsize=8)
draw_box(ax, 8.0, box_y, 3.8, 0.55, 'missense_variants.tsv\nWT/Mut protein sequences', '#27ae60', fontsize=8)
draw_box(ax, 11.8, box_y, 3.0, 0.55, 'dna_variants.tsv\n7,954 w/ nucleotide ctx', '#c0392b', fontsize=8)

arrow_split(ax, 7, sec_y - 0.25, [2.8, 8.0, 11.8], box_y + 0.275)
ax.annotate('', xy=(4.8, box_y), xytext=(6.1, box_y),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
ax.annotate('', xy=(10.3, box_y), xytext=(9.9, box_y),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

# ===== SECTION 3: MODEL LOADING =====
sec_y = 16.8
draw_box(ax, 7, sec_y, 5, 0.5, 'MODEL LOADING & SETUP', '#2c3e50', fontsize=12, bold=True)
arrow_down(ax, 7, 17.375, sec_y + 0.25)

box_y = 15.95
draw_box(ax, 3.8, box_y, 5.5, 0.55, 'PROTEIN MODELS\nESM1b (650M) | ESM2 (3B) | ESM3 (1.4B) | ProtT5 (3B)', '#8e44ad', fontsize=8, bold=True)
draw_box(ax, 10.5, box_y, 5.5, 0.55, 'DNA MODELS\nDNABERT-2 (117M) | HyenaDNA (~1B) | NT-v2 (500M)', '#c0392b', fontsize=8, bold=True)

arrow_split(ax, 7, sec_y - 0.25, [3.8, 10.5], box_y + 0.275)

# ===== SECTION 4: EMBEDDING =====
sec_y = 15.1
draw_box(ax, 7, sec_y, 5, 0.5, 'EMBEDDING EXTRACTION', '#2c3e50', fontsize=12, bold=True)
arrow_down(ax, 3.8, box_y - 0.275, sec_y + 0.25, '#8e44ad')
arrow_down(ax, 10.5, box_y - 0.275, sec_y + 0.25, '#c0392b')

box_y = 14.25
draw_box(ax, 3.8, box_y, 5.5, 0.55, 'WT & Mut embeddings at variant pos\nLayer 15 (ESM1b/ESM2) | Final (ESM3/ProtT5)', '#2980b9', fontsize=8)
draw_box(ax, 10.5, box_y, 5.5, 0.55, '~6002bp nucleotide context\nDNABERT-2 (BPE) | HyenaDNA (1bp) | NT-v2 (6-mer)', '#e74c3c', fontsize=8)

# ===== SECTION 5: SCORING =====
sec_y = 13.4
draw_box(ax, 7, sec_y, 5, 0.5, 'SCORING', '#2c3e50', fontsize=12, bold=True)
arrow_down(ax, 7, 13.95, sec_y + 0.25)

# Merge arrows from both embedding boxes into scoring
arrow_merge(ax, 7, sec_y + 0.25, [3.8, 10.5], box_y - 0.275)

box_y = 12.55
draw_box(ax, 3.8, box_y, 5.5, 0.55, 'Cosine Distance\n1 - cos_sim(WT, Mut)  [7 models]', '#3498db', fontsize=8)
draw_box(ax, 10.5, box_y, 5.5, 0.55, 'Masked Marginal Log-Likelihood\nP(wt) - P(mut)  [ESM1b-MLM]', '#9b59b6', fontsize=8)

arrow_split(ax, 7, sec_y - 0.25, [3.8, 10.5], box_y + 0.275)

# ===== SECTION 6: EVALUATION =====
sec_y = 11.7
draw_box(ax, 7, sec_y, 5, 0.5, 'EVALUATION', '#2c3e50', fontsize=12, bold=True)
arrow_down(ax, 7, box_y - 0.275, sec_y + 0.25)

box_y = 10.85
draw_box(ax, 2.2, box_y, 3.2, 0.55, 'AUROC | AUPRC | F1\nAccuracy | Precision | Recall', '#27ae60', fontsize=8)
draw_box(ax, 5.8, box_y, 3.0, 0.55, 'Mann-Whitney U\nSpearman Correlation', '#f39c12', fontsize=8)
draw_box(ax, 9.2, box_y, 3.0, 0.55, 'ROC Curves\nPR Curves', '#1abc9c', fontsize=8)
draw_box(ax, 12.3, box_y, 2.6, 0.55, 'Distributions\nBoxplots', '#e67e22', fontsize=8)

arrow_split(ax, 7, sec_y - 0.25, [2.2, 5.8, 9.2, 12.3], box_y + 0.275)

# ===== SECTION 7: PLOTTING & OUTPUT =====
sec_y = 10.0
draw_box(ax, 7, sec_y, 5, 0.5, 'PLOTTING & OUTPUT', '#2c3e50', fontsize=12, bold=True)
arrow_down(ax, 7, box_y - 0.275, sec_y + 0.25)

box_y = 9.15
draw_box(ax, 2.5, box_y, 3.8, 0.55, 'dataset_analysis.png\nClass dist | Gene counts\nAA substitution matrix', '#34495e', fontsize=7)
draw_box(ax, 7.0, box_y, 3.5, 0.55, 'benchmark_summary.csv\n8 models × 12 metrics', '#34495e', fontsize=7)
draw_box(ax, 11.0, box_y, 4.0, 0.55, 'roc_curves | pr_curves\nauroc_comparison | distributions', '#34495e', fontsize=7)

arrow_split(ax, 7, sec_y - 0.25, [2.5, 7.0, 11.0], box_y + 0.275)

# ===== RESULTS BOX =====
sec_y = 8.3
draw_box(ax, 7, sec_y, 5, 0.5, 'FINAL RESULTS', '#2c3e50', fontsize=12, bold=True)
arrow_down(ax, 7, box_y - 0.275, sec_y + 0.25)

# Results table
res_y = 7.1
draw_box(ax, 7, res_y, 12, 0.95, '', '#f8f9fa', alpha=0.7)

results_lines = [
    ("ESM1b-MLM", "0.889", "#2ecc71"),
    ("ESM3", "0.759", "#27ae60"),
    ("ESM1b", "0.587", "#3498db"),
    ("ESM2", "0.530", "#3498db"),
    ("ProtT5", "0.529", "#3498db"),
    ("NT-v2", "0.514", "#e67e22"),
    ("DNABERT-2", "0.491", "#e74c3c"),
    ("HyenaDNA", "0.481", "#e74c3c"),
]

x_start = 1.8
for i, (name, auroc, color) in enumerate(results_lines):
    x = x_start + i * 1.5
    ax.add_patch(FancyBboxPatch((x - 0.55, 6.85), 1.1, 0.5, boxstyle='round,pad=0.03',
                 facecolor=color, edgecolor='#555', linewidth=0.8, alpha=0.85, zorder=2))
    ax.text(x, 7.05, f'{name}', ha='center', fontsize=7, fontweight='bold', color='white', zorder=3)
    ax.text(x, 6.93, auroc, ha='center', fontsize=6.5, color='white', zorder=3)

ax.text(7, 6.55, 'Best: ESM1b-MLM (Masked Marginal Log-Likelihood) — far outperforms all cosine distance methods',
        ha='center', fontsize=9, color='#2c3e50', fontweight='bold')

arrow_down(ax, 7, sec_y - 0.25, 7.6)

# Infrastructure footer
ax.text(7, 6.05, 'PBS cluster (gnode6) | 8× A100-SXM4-40GB | CUDA 12.4 | PyTorch 2.4',
        ha='center', fontsize=8, color='#aaaaaa')

plt.tight_layout()
plt.savefig('figures/project_pipeline.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved figures/project_pipeline.png")
