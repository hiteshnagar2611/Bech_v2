import pickle, csv, torch, numpy as np
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score

variants = []
with open('missense_variants.tsv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        variants.append(row)

import esm
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model = model.eval().cuda()
batch_converter = alphabet.get_batch_converter()

protein_groups = defaultdict(list)
for v in variants:
    protein_groups[v['refseq_accession']].append(v)

test_variants = []
for prot_id, pvars in list(protein_groups.items()):
    test_variants.extend(pvars[:3])
print(f"Testing {len(test_variants)} variants across layers...")

for layer in [6, 12, 15, 20, 25, 30, 33]:
    all_dists = []
    all_labels = []

    for prot_id, pvars in protein_groups.items():
        test_pvars = [v for v in pvars if v in test_variants]
        if not test_pvars:
            continue

        wt_seq = test_pvars[0]['wt_seq']
        data = [("wt", wt_seq)]
        _, _, wt_tokens = batch_converter(data)
        wt_tokens = wt_tokens.to('cuda')
        with torch.no_grad():
            wt_out = model(wt_tokens, repr_layers=[layer], return_contacts=False)
        wt_emb = wt_out["representations"][layer][0].cpu()

        mut_seqs = [v['mut_seq'] for v in test_pvars]
        positions = [int(v['protein_position']) - 1 for v in test_pvars]

        bs = 32
        for bstart in range(0, len(mut_seqs), bs):
            batch = mut_seqs[bstart:bstart+bs]
            batch_pos = positions[bstart:bstart+bs]
            batch_vars = test_pvars[bstart:bstart+bs]
            data = [(f"m{i}", s) for i, s in enumerate(batch)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to('cuda')
            with torch.no_grad():
                out = model(tokens, repr_layers=[layer], return_contacts=False)
            mut_emb = out["representations"][layer].cpu()

            for i, (pos, v) in enumerate(zip(batch_pos, batch_vars)):
                tok_pos = pos + 1
                if tok_pos < wt_emb.shape[0] and tok_pos < mut_emb.shape[1]:
                    wt_vec = wt_emb[tok_pos].unsqueeze(0)
                    mut_vec = mut_emb[i, tok_pos].unsqueeze(0)
                    cos_sim = F.cosine_similarity(wt_vec, mut_vec).item()
                    all_dists.append(1.0 - cos_sim)
                    all_labels.append(1 if v['clinical_significance'] == 'Pathogenic' else 0)

    y_true = np.array(all_labels)
    y_score = np.array(all_dists)
    auroc = roc_auc_score(y_true, y_score)
    auroc_inv = roc_auc_score(y_true, -y_score)
    path_d = y_score[y_true == 1].mean()
    ben_d = y_score[y_true == 0].mean()
    print(f"  Layer {layer:2d}: AUROC={auroc:.4f}  AUROC_inv={auroc_inv:.4f}  "
          f"Path_mean={path_d:.6f}  Ben_mean={ben_d:.6f}  N={len(y_true)}")

del model
torch.cuda.empty_cache()
