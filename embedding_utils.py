import torch
import torch.nn.functional as F
import numpy as np


def cosine_distance(vec_a, vec_b):
    a = torch.tensor(vec_a, dtype=torch.float32) if not isinstance(vec_a, torch.Tensor) else vec_a.float()
    b = torch.tensor(vec_b, dtype=torch.float32) if not isinstance(vec_b, torch.Tensor) else vec_b.float()
    cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return 1.0 - cos_sim


def load_esm_model(model_name='esm1b'):
    import esm
    if model_name == 'esm1b':
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    elif model_name == 'esm2':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.eval()
    return model, alphabet


def batch_embed_esm(model, alphabet, sequences, device='cuda', repr_layer=33):
    batch_converter = alphabet.get_batch_converter()
    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    embeddings = results["representations"][repr_layer]
    return embeddings.cpu()


def extract_batch_positions_esm(model, alphabet, wt_seq, mut_seqs, variant_positions, device='cuda', batch_size=32):
    all_sequences = [wt_seq] + mut_seqs
    all_embeddings = []

    for i in range(0, len(all_sequences), batch_size):
        batch = all_sequences[i:i + batch_size]
        emb = batch_embed_esm(model, alphabet, batch, device=device)
        all_embeddings.append(emb)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    wt_embedding = all_embeddings[0]
    mut_embeddings = all_embeddings[1:]

    wt_vecs = []
    mut_vecs = []
    for idx, pos in enumerate(variant_positions):
        wt_vecs.append(wt_embedding[pos + 1])
        mut_vecs.append(mut_embeddings[idx, pos + 1])

    return torch.stack(wt_vecs), torch.stack(mut_vecs)


def compute_cosine_distances(wt_vecs, mut_vecs):
    wt_norm = F.normalize(wt_vecs, dim=1)
    mut_norm = F.normalize(mut_vecs, dim=1)
    cos_sim = torch.sum(wt_norm * mut_norm, dim=1)
    return (1.0 - cos_sim).numpy()


def load_esm3_model():
    from esm.models.esm3 import ESM3
    model = ESM3.from_pretrained("esm3-open")
    model = model.eval()
    return model


def extract_batch_positions_esm3(model, wt_seq, mut_seqs, variant_positions, device='cuda', batch_size=8):
    from esm.sdk.api import ESMProtein

    def get_embedding(model, sequence, device):
        protein = ESMProtein(sequence=sequence)
        with torch.no_grad():
            output = model(protein)
        return output.protein_function_class_logits if hasattr(output, 'protein_function_class_logits') else output

    wt_emb = get_embedding(model, wt_seq, device)

    mut_embs = []
    for i in range(0, len(mut_seqs), batch_size):
        batch = mut_seqs[i:i + batch_size]
        for seq in batch:
            emb = get_embedding(model, seq, device)
            mut_embs.append(emb)

    wt_vecs = []
    mut_vecs = []
    for idx, pos in enumerate(variant_positions):
        wt_vecs.append(wt_emb[pos])
        mut_vecs.append(mut_embs[idx][pos])

    return torch.stack(wt_vecs), torch.stack(mut_vecs)
