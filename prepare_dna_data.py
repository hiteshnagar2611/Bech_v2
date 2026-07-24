import gzip
import re
import csv
from collections import Counter

AA_3_TO_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}

MISSENSE_RE = re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
STOP_CODONS = {'Ter', 'Amber', 'Ochre', 'Opal'}
CENTER = 3001

infile = 'clinvar_rcv_enriched.tsv.gz'
outfile = 'dna_variants.tsv'

fieldnames = [
    'rcv_accession', 'gene_symbol', 'variant_position',
    'nuc_context_wt', 'nuc_context_mut',
    'label', 'wt_aa', 'mut_aa', 'protein_position',
]

stats = Counter()
rows = []

with gzip.open(infile, 'rt') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        stats['total'] += 1

        pc = row.get('protein_change', '').strip()
        wt_seq = row.get('aa_seq_wt_full', '').strip()
        mut_seq = row.get('aa_seq_mut_full', '').strip()
        sig = row.get('clinical_significance', '').strip()
        wt_nuc = row.get('nuc_context_wt', '').strip()
        mut_nuc = row.get('nuc_context_mut', '').strip()
        ref_allele = row.get('ref_allele_vcf', '').strip()
        alt_allele = row.get('alt_allele_vcf', '').strip()

        if not wt_seq or not mut_seq:
            stats['empty_sequence'] += 1
            continue

        m = MISSENSE_RE.match(pc)
        if not m:
            stats['non_missense'] += 1
            continue

        wt_3, pos_str, mut_3 = m.group(1), m.group(2), m.group(3)
        if wt_3 in STOP_CODONS or mut_3 in STOP_CODONS:
            stats['non_missense'] += 1
            continue

        wt_aa = AA_3_TO_1.get(wt_3)
        mut_aa = AA_3_TO_1.get(mut_3)
        if wt_aa is None or mut_aa is None:
            stats['non_missense'] += 1
            continue
        if wt_aa == mut_aa:
            stats['non_missense'] += 1
            continue

        position = int(pos_str)
        if position > len(wt_seq):
            stats['position_out_of_range'] += 1
            continue
        if wt_seq[position - 1] != wt_aa:
            stats['wt_mismatch'] += 1
            continue

        if not wt_nuc or not mut_nuc:
            stats['empty_nuc_context'] += 1
            continue

        diffs = [i for i in range(min(len(wt_nuc), len(mut_nuc))) if wt_nuc[i] != mut_nuc[i]]

        if len(diffs) == 0:
            mut_nuc = list(wt_nuc)
            mut_nuc[CENTER] = alt_allele
            mut_nuc = ''.join(mut_nuc)
            stats['fixed_no_diff'] += 1
        elif len(diffs) == 1 and diffs[0] == CENTER:
            stats['clean_single_diff'] += 1
        else:
            stats['skipped_indel'] += 1
            continue

        rows.append({
            'rcv_accession': row.get('rcv_accession', ''),
            'gene_symbol': row.get('gene_symbol', ''),
            'variant_position': CENTER,
            'nuc_context_wt': wt_nuc,
            'nuc_context_mut': mut_nuc,
            'label': sig,
            'wt_aa': wt_aa,
            'mut_aa': mut_aa,
            'protein_position': position,
        })

with open(outfile, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

print(f"Output: {outfile} ({len(rows)} variants)")
print(f"\nStats:")
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")

labels = Counter(r['label'] for r in rows)
print(f"\nLabel distribution:")
for k, v in sorted(labels.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

genes = Counter(r['gene_symbol'] for r in rows)
print(f"\nUnique genes: {len(genes)}")
