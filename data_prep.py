import gzip
import re
import csv
import sys
from collections import Counter

AA_3_TO_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}

MISSENSE_RE = re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
STOP_CODONS = {'Ter', 'Amber', 'Ochre', 'Opal'}


def parse_protein_change(pc):
    m = MISSENSE_RE.match(pc)
    if not m:
        return None
    wt_3, pos_str, mut_3 = m.group(1), m.group(2), m.group(3)
    if wt_3 in STOP_CODONS or mut_3 in STOP_CODONS:
        return None
    wt_1 = AA_3_TO_1.get(wt_3)
    mut_1 = AA_3_TO_1.get(mut_3)
    if wt_1 is None or mut_1 is None:
        return None
    if wt_1 == mut_1:
        return None
    return wt_1, int(pos_str), mut_1


def main():
    infile = 'clinvar_rcv_enriched.tsv.gz'
    outfile = 'missense_variants.tsv'

    fieldnames = [
        'rcv_accession', 'gene_symbol', 'protein_position',
        'wt_aa', 'mut_aa', 'refseq_accession', 'wt_seq', 'mut_seq',
        'clinical_significance'
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

            if not wt_seq or not mut_seq:
                stats['empty_sequence'] += 1
                continue

            parsed = parse_protein_change(pc)
            if parsed is None:
                stats['non_missense'] += 1
                continue

            wt_aa, position, mut_aa = parsed

            if position > len(wt_seq):
                stats['position_out_of_range'] += 1
                continue

            wt_aa_in_seq = wt_seq[position - 1]
            if wt_aa_in_seq != wt_aa:
                stats['wt_mismatch'] += 1
                continue

            rows.append({
                'rcv_accession': row.get('rcv_accession', ''),
                'gene_symbol': row.get('gene_symbol', ''),
                'protein_position': position,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'refseq_accession': row.get('refseq_protein_accession', ''),
                'wt_seq': wt_seq,
                'mut_seq': mut_seq,
                'clinical_significance': sig,
            })
            stats['missense'] += 1

    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    print(f"Output: {outfile} ({len(rows)} variants)")
    print(f"\nStats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    labels = Counter(r['clinical_significance'] for r in rows)
    print(f"\nLabel distribution:")
    for k, v in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    genes = Counter(r['gene_symbol'] for r in rows)
    print(f"\nUnique proteins (refseq): {len(set(r['refseq_accession'] for r in rows))}")
    print(f"Unique genes: {len(genes)}")


if __name__ == '__main__':
    main()
