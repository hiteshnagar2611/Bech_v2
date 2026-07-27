import argparse
import csv
import os
import pickle
import time
from collections import defaultdict

import torch
import numpy as np


def load_variants(tsfile):
    variants = []
    with open(tsfile, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            variants.append(row)
    return variants


def group_by_protein(variants):
    groups = defaultdict(list)
    for v in variants:
        groups[v['refseq_accession']].append(v)
    return groups


def get_batch_size(seq_len):
    if seq_len < 300:
        return 64
    elif seq_len < 600:
        return 32
    else:
        return 16


def run_esm(model_name, variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from embedding_utils import load_esm_model, batch_embed_esm

    print(f"Loading {model_name}...")
    model, alphabet = load_esm_model(model_name)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = 15

    protein_groups = group_by_protein(variants)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, f'{model_name}_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 500

    for prot_idx, (protein_id, prot_variants) in enumerate(protein_groups.items()):
        remaining = [v for v in prot_variants if v['rcv_accession'] not in done_rcvs]
        if not remaining:
            continue

        wt_seq = remaining[0]['wt_seq']
        positions = [int(v['protein_position']) - 1 for v in remaining]
        mut_seqs = [v['mut_seq'] for v in remaining]

        # Embed WT once
        data = [("wt", wt_seq)]
        _, _, wt_tokens = batch_converter(data)
        wt_tokens = wt_tokens.to(device)
        with torch.no_grad():
            wt_results = model(wt_tokens, repr_layers=[repr_layer], return_contacts=False)
        wt_embedding = wt_results["representations"][repr_layer][0].cpu()

        # Batch mutant sequences
        max_len = max(len(s) for s in mut_seqs) if mut_seqs else len(wt_seq)
        bs = batch_size_override or get_batch_size(max_len)

        for batch_start in range(0, len(mut_seqs), bs):
            batch_mut_seqs = mut_seqs[batch_start:batch_start + bs]
            batch_positions = positions[batch_start:batch_start + bs]
            batch_variants = remaining[batch_start:batch_start + bs]

            data = [(f"mut_{i}", s) for i, s in enumerate(batch_mut_seqs)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results_dict = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            mut_embeddings = results_dict["representations"][repr_layer].cpu()

            wt_vecs = []
            mut_vecs = []
            valid_variants = []
            for i, (pos, v) in enumerate(zip(batch_positions, batch_variants)):
                tok_pos = pos + 1
                if tok_pos < wt_embedding.shape[0] and tok_pos < mut_embeddings.shape[1]:
                    wt_vecs.append(wt_embedding[tok_pos])
                    mut_vecs.append(mut_embeddings[i, tok_pos])
                    valid_variants.append(v)

            if wt_vecs:
                wt_tensor = torch.stack(wt_vecs)
                mut_tensor = torch.stack(mut_vecs)

                wt_norm = F.normalize(wt_tensor, dim=1)
                mut_norm = F.normalize(mut_tensor, dim=1)
                cos_sim = torch.sum(wt_norm * mut_norm, dim=1)
                distances = (1.0 - cos_sim).numpy()

                for v, d in zip(valid_variants, distances):
                    results.append({
                        'rcv_accession': v['rcv_accession'],
                        'gene_symbol': v['gene_symbol'],
                        'protein_position': v['protein_position'],
                        'wt_aa': v['wt_aa'],
                        'mut_aa': v['mut_aa'],
                        'cosine_distance': float(d),
                        'label': v['clinical_significance'],
                    })
                    processed += 1

            done_rcvs.update(v['rcv_accession'] for v in batch_variants)

        if processed % save_interval < bs or prot_idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [{model_name}] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [{model_name}] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def run_esm_mlm(model_name, variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from embedding_utils import load_esm_model

    print(f"Loading {model_name} for masked marginal log-likelihood...")
    model, alphabet = load_esm_model(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    for p in model.parameters():
        p.requires_grad = False

    mask_idx = alphabet.mask_idx
    print(f"  Mask token ID: {mask_idx}")

    protein_groups = group_by_protein(variants)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_mlm_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, f'{model_name}_mlm_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 500

    for prot_idx, (protein_id, prot_variants) in enumerate(protein_groups.items()):
        remaining = [v for v in prot_variants if v['rcv_accession'] not in done_rcvs]
        if not remaining:
            continue

        wt_seq = remaining[0]['wt_seq']
        data = [("wt", wt_seq)]
        _, _, wt_tokens = batch_converter(data)
        wt_tokens = wt_tokens.to(device)
        seq_len = wt_tokens.shape[1]

        for v in remaining:
            pos = int(v['protein_position']) - 1
            tok_pos = pos + 1
            wt_aa = v['wt_aa']
            mut_aa = v['mut_aa']

            if tok_pos >= seq_len:
                continue

            masked_tokens = wt_tokens.clone()
            masked_tokens[0, tok_pos] = mask_idx

            with torch.no_grad():
                results_dict = model(masked_tokens, repr_layers=[], return_contacts=False)
            logits = results_dict["logits"][0].cpu()

            logits_at_pos = logits[tok_pos]
            probs = F.softmax(logits_at_pos, dim=0)

            wt_idx = alphabet.get_idx(wt_aa)
            mut_idx = alphabet.get_idx(mut_aa)

            ll_wt = float(probs[wt_idx])
            ll_mut = float(probs[mut_idx])

            score = ll_wt - ll_mut

            results.append({
                'rcv_accession': v['rcv_accession'],
                'gene_symbol': v['gene_symbol'],
                'protein_position': v['protein_position'],
                'wt_aa': v['wt_aa'],
                'mut_aa': v['mut_aa'],
                'cosine_distance': float(score),
                'label': v['clinical_significance'],
            })
            processed += 1
            done_rcvs.add(v['rcv_accession'])

        if processed % save_interval < len(remaining) or prot_idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [{model_name}_mlm] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [{model_name}_mlm] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def run_esm3_model(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F

    print("Loading ESM3-open...")
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    model = ESM3.from_pretrained("esm3-open", device=torch.device("cpu"))
    model = model.to(device)
    model = model.eval()

    for p in model.parameters():
        p.requires_grad = False

    def esm3_embed(sequence):
        protein = ESMProtein(sequence=sequence)
        encoded = model.encode(protein)
        forward_input = {}
        field_map = {
            "sequence": "sequence_tokens",
            "structure": "structure_tokens",
            "secondary_structure": "ss8_tokens",
            "sasa": "sasa_tokens",
            "function": "function_tokens",
            "residue_annotations": "residue_annotation_tokens",
        }
        for src, dst in field_map.items():
            val = getattr(encoded, src, None)
            if val is not None and isinstance(val, torch.Tensor):
                if val.dim() == 1:
                    val = val.unsqueeze(0)
                forward_input[dst] = val.to(device)
        with torch.no_grad():
            output = model.forward(**forward_input)
        return output.embeddings[0].cpu().float()

    protein_groups = group_by_protein(variants)
    checkpoint_path = os.path.join(checkpoint_dir, 'esm3_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'esm3_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()

    for protein_id, prot_variants in protein_groups.items():
        remaining = [v for v in prot_variants if v['rcv_accession'] not in done_rcvs]
        if not remaining:
            continue

        wt_seq = remaining[0]['wt_seq']
        wt_embed = esm3_embed(wt_seq)

        for v in remaining:
            pos = int(v['protein_position']) - 1
            mut_seq = v['mut_seq']
            mut_embed = esm3_embed(mut_seq)

            tok_pos = pos + 1
            if tok_pos < wt_embed.shape[0] and tok_pos < mut_embed.shape[0]:
                wt_vec = wt_embed[tok_pos].unsqueeze(0).float()
                mut_vec = mut_embed[tok_pos].unsqueeze(0).float()
                cos_sim = F.cosine_similarity(wt_vec, mut_vec).item()
                dist = 1.0 - cos_sim
            else:
                dist = float('nan')

            results.append({
                'rcv_accession': v['rcv_accession'],
                'gene_symbol': v['gene_symbol'],
                'protein_position': v['protein_position'],
                'wt_aa': v['wt_aa'],
                'mut_aa': v['mut_aa'],
                'cosine_distance': float(dist),
                'label': v['clinical_significance'],
            })
            processed += 1
            done_rcvs.add(v['rcv_accession'])

        if processed % 500 < len(remaining):
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [esm3] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [esm3] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def run_prott5(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from transformers import T5EncoderModel, T5Tokenizer

    model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print(f"Loading {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    def tokenize_seqs(seqs, max_length=1024):
        spaced = [" ".join(seq) for seq in seqs]
        encoded = tokenizer(
            spaced,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded.input_ids.to(device), encoded.attention_mask.to(device)

    protein_groups = group_by_protein(variants)
    checkpoint_path = os.path.join(checkpoint_dir, 'prott5_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'prott5_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 500

    for prot_idx, (protein_id, prot_variants) in enumerate(protein_groups.items()):
        remaining = [v for v in prot_variants if v['rcv_accession'] not in done_rcvs]
        if not remaining:
            continue

        wt_seq = remaining[0]['wt_seq']
        mut_seqs = [v['mut_seq'] for v in remaining]

        with torch.no_grad():
            wt_ids, wt_mask = tokenize_seqs([wt_seq])
            wt_out = model(input_ids=wt_ids, attention_mask=wt_mask)
            wt_embedding = wt_out.last_hidden_state[0].cpu().float()

        max_len = max(len(s) for s in mut_seqs) if mut_seqs else len(wt_seq)
        bs = batch_size_override or get_batch_size(max_len)

        for batch_start in range(0, len(mut_seqs), bs):
            batch_mut_seqs = mut_seqs[batch_start:batch_start + bs]
            batch_variants = remaining[batch_start:batch_start + bs]

            with torch.no_grad():
                mut_ids, mut_mask = tokenize_seqs(batch_mut_seqs)
                mut_out = model(input_ids=mut_ids, attention_mask=mut_mask)
                mut_embeddings = mut_out.last_hidden_state.cpu().float()

            wt_vecs = []
            mut_vecs = []
            valid_variants = []
            for i, v in enumerate(batch_variants):
                pos = int(v['protein_position']) - 1
                tok_pos = pos + 1
                if tok_pos < wt_embedding.shape[0] and tok_pos < mut_embeddings.shape[1]:
                    wt_vecs.append(wt_embedding[tok_pos])
                    mut_vecs.append(mut_embeddings[i, tok_pos])
                    valid_variants.append(v)

            if wt_vecs:
                wt_tensor = torch.stack(wt_vecs)
                mut_tensor = torch.stack(mut_vecs)
                cos_sim = F.cosine_similarity(wt_tensor, mut_tensor)
                distances = (1.0 - cos_sim).numpy()

                for v, d in zip(valid_variants, distances):
                    results.append({
                        'rcv_accession': v['rcv_accession'],
                        'gene_symbol': v['gene_symbol'],
                        'protein_position': v['protein_position'],
                        'wt_aa': v['wt_aa'],
                        'mut_aa': v['mut_aa'],
                        'cosine_distance': float(d),
                        'label': v['clinical_significance'],
                    })
                    processed += 1

            done_rcvs.update(v['rcv_accession'] for v in batch_variants)

        if processed % save_interval < bs or prot_idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [prott5] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [prott5] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def run_dnabert2(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    import sys
    import os
    import importlib
    from transformers import PreTrainedTokenizerFast, AutoConfig
    from huggingface_hub import snapshot_download

    model_id = "zhihan1996/DNABERT-2-117M"
    print(f"Loading {model_id}...")

    model_dir = snapshot_download(model_id)
    init_path = os.path.join(model_dir, '__init__.py')
    if not os.path.exists(init_path):
        open(init_path, 'w').close()
    parent = os.path.dirname(model_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    pkg_name = os.path.basename(model_dir)
    mod = importlib.import_module(f'{pkg_name}.bert_layers')
    mod.flash_attn_qkvpacked_func = None
    CustomBertModel = mod.BertModel

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = CustomBertModel.from_pretrained(model_id, config=config)
    model.config.output_hidden_states = True
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    max_tokens = 512
    var_center_bp = 3001
    bs = batch_size_override or 8

    def tokenize_and_find_var(seq):
        encoded = tokenizer(seq, return_tensors="pt", truncation=False, return_offsets_mapping=True)
        offsets = encoded.offset_mapping[0]
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        n_tokens = input_ids.shape[1]

        var_token_idx = 0
        for i, (start, end) in enumerate(offsets):
            if start <= var_center_bp < end:
                var_token_idx = i
                break

        if n_tokens > max_tokens:
            half = max_tokens // 2
            tok_start = max(0, var_token_idx - half)
            tok_end = min(n_tokens, tok_start + max_tokens)
            if tok_end == n_tokens:
                tok_start = tok_end - max_tokens
            input_ids = input_ids[:, tok_start:tok_end]
            attention_mask = attention_mask[:, tok_start:tok_end]
            var_token_idx = var_token_idx - tok_start

        return input_ids, attention_mask, var_token_idx

    checkpoint_path = os.path.join(checkpoint_dir, 'dnabert2_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'dnabert2_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 200

    remaining = [v for v in variants if v['rcv_accession'] not in done_rcvs]

    for batch_start in range(0, len(remaining), bs):
        batch = remaining[batch_start:batch_start + bs]
        wt_seqs = [v['nuc_context_wt'] for v in batch]
        mut_seqs = [v['nuc_context_mut'] for v in batch]

        wt_ids_list, wt_masks_list, wt_var_indices = [], [], []
        for seq in wt_seqs:
            ids, mask, idx = tokenize_and_find_var(seq)
            wt_ids_list.append(ids)
            wt_masks_list.append(mask)
            wt_var_indices.append(idx)

        mut_ids_list, mut_masks_list, mut_var_indices = [], [], []
        for seq in mut_seqs:
            ids, mask, idx = tokenize_and_find_var(seq)
            mut_ids_list.append(ids)
            mut_masks_list.append(mask)
            mut_var_indices.append(idx)

        with torch.no_grad():
            wt_embs = []
            for ids, mask in zip(wt_ids_list, wt_masks_list):
                result = model(input_ids=ids, attention_mask=mask)
                emb = result[0][0].cpu().float()
                wt_embs.append(emb)

            mut_embs = []
            for ids, mask in zip(mut_ids_list, mut_masks_list):
                result = model(input_ids=ids, attention_mask=mask)
                emb = result[0][0].cpu().float()
                mut_embs.append(emb)

        wt_vecs, mut_vecs, valid = [], [], []
        for i, v in enumerate(batch):
            widx = wt_var_indices[i]
            midx = mut_var_indices[i]
            if widx < wt_embs[i].shape[0] and midx < mut_embs[i].shape[0]:
                wt_vecs.append(wt_embs[i][widx])
                mut_vecs.append(mut_embs[i][midx])
                valid.append(v)

        if wt_vecs:
            wt_tensor = torch.stack(wt_vecs)
            mut_tensor = torch.stack(mut_vecs)
            cos_sim = F.cosine_similarity(wt_tensor, mut_tensor)
            distances = (1.0 - cos_sim).numpy()

            for v, d in zip(valid, distances):
                results.append({
                    'rcv_accession': v['rcv_accession'],
                    'gene_symbol': v['gene_symbol'],
                    'protein_position': v['protein_position'],
                    'wt_aa': v['wt_aa'],
                    'mut_aa': v['mut_aa'],
                    'cosine_distance': float(d),
                    'label': v['label'],
                })
                processed += 1

        done_rcvs.update(v['rcv_accession'] for v in batch)

        if processed % save_interval < bs:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [dnabert2] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [dnabert2] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def run_hyenadna(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    model_id = "LongSafari/hyenadna-large-1m-seqlen-hf"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    var_center = 3001
    bs = batch_size_override or 4

    checkpoint_path = os.path.join(checkpoint_dir, 'hyenadna_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'hyenadna_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 200

    remaining = [v for v in variants if v['rcv_accession'] not in done_rcvs]

    for batch_start in range(0, len(remaining), bs):
        batch = remaining[batch_start:batch_start + bs]
        wt_seqs = [v['nuc_context_wt'] for v in batch]
        mut_seqs = [v['nuc_context_mut'] for v in batch]

        wt_encoded = tokenizer(wt_seqs, return_tensors="pt", padding=True, truncation=False)
        wt_input = wt_encoded.input_ids.to(device)

        mut_encoded = tokenizer(mut_seqs, return_tensors="pt", padding=True, truncation=False)
        mut_input = mut_encoded.input_ids.to(device)

        with torch.no_grad():
            wt_out = model(input_ids=wt_input)
            wt_hidden = wt_out.last_hidden_state.cpu().float()

            mut_out = model(input_ids=mut_input)
            mut_hidden = mut_out.last_hidden_state.cpu().float()

        wt_vecs, mut_vecs, valid = [], [], []
        for i, v in enumerate(batch):
            if var_center < wt_hidden.shape[1] and var_center < mut_hidden.shape[1]:
                wt_vecs.append(wt_hidden[i, var_center])
                mut_vecs.append(mut_hidden[i, var_center])
                valid.append(v)

        if wt_vecs:
            wt_tensor = torch.stack(wt_vecs)
            mut_tensor = torch.stack(mut_vecs)
            cos_sim = F.cosine_similarity(wt_tensor, mut_tensor)
            distances = (1.0 - cos_sim).numpy()

            for v, d in zip(valid, distances):
                results.append({
                    'rcv_accession': v['rcv_accession'],
                    'gene_symbol': v['gene_symbol'],
                    'protein_position': v['protein_position'],
                    'wt_aa': v['wt_aa'],
                    'mut_aa': v['mut_aa'],
                    'cosine_distance': float(d),
                    'label': v['label'],
                })
                processed += 1

        done_rcvs.update(v['rcv_accession'] for v in batch)

        if processed % save_interval < bs:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [hyenadna] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [hyenadna] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def run_ntv2(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    model_id = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    full_model = AutoModelForTokenClassification.from_pretrained(model_id, trust_remote_code=True)
    esm_model = full_model.esm
    esm_model = esm_model.to(device).eval()
    del full_model

    for p in esm_model.parameters():
        p.requires_grad = False

    var_center = 3001
    kmer = 6
    var_token_idx = var_center // kmer
    bs = batch_size_override or 4

    checkpoint_path = os.path.join(checkpoint_dir, 'ntv2_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'ntv2_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 200

    remaining = [v for v in variants if v['rcv_accession'] not in done_rcvs]

    for batch_start in range(0, len(remaining), bs):
        batch = remaining[batch_start:batch_start + bs]
        wt_seqs = [v['nuc_context_wt'] for v in batch]
        mut_seqs = [v['nuc_context_mut'] for v in batch]

        wt_encoded = tokenizer(wt_seqs, return_tensors="pt", padding=True, truncation=False)
        wt_input = wt_encoded.input_ids.to(device)
        wt_mask = wt_encoded.attention_mask.to(device)

        mut_encoded = tokenizer(mut_seqs, return_tensors="pt", padding=True, truncation=False)
        mut_input = mut_encoded.input_ids.to(device)
        mut_mask = mut_encoded.attention_mask.to(device)

        with torch.no_grad():
            wt_out = esm_model(input_ids=wt_input, attention_mask=wt_mask)
            wt_hidden = wt_out.last_hidden_state.cpu().float()

            mut_out = esm_model(input_ids=mut_input, attention_mask=mut_mask)
            mut_hidden = mut_out.last_hidden_state.cpu().float()

        wt_vecs, mut_vecs, valid = [], [], []
        for i, v in enumerate(batch):
            if var_token_idx < wt_hidden.shape[1] and var_token_idx < mut_hidden.shape[1]:
                wt_vecs.append(wt_hidden[i, var_token_idx])
                mut_vecs.append(mut_hidden[i, var_token_idx])
                valid.append(v)

        if wt_vecs:
            wt_tensor = torch.stack(wt_vecs)
            mut_tensor = torch.stack(mut_vecs)
            cos_sim = F.cosine_similarity(wt_tensor, mut_tensor)
            distances = (1.0 - cos_sim).numpy()

            for v, d in zip(valid, distances):
                results.append({
                    'rcv_accession': v['rcv_accession'],
                    'gene_symbol': v['gene_symbol'],
                    'protein_position': v['protein_position'],
                    'wt_aa': v['wt_aa'],
                    'mut_aa': v['mut_aa'],
                    'cosine_distance': float(d),
                    'label': v['label'],
                })
                processed += 1

        done_rcvs.update(v['rcv_accession'] for v in batch)

        if processed % save_interval < bs:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [ntv2] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [ntv2] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del esm_model
    torch.cuda.empty_cache()
    return results_path


def run_protbert(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    model_name = "Rostlab/prot_bert_bfd"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    bs = batch_size_override or get_batch_size(1024)

    protein_groups = group_by_protein(variants)
    checkpoint_path = os.path.join(checkpoint_dir, 'protbert_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'protbert_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 500

    for prot_idx, (protein_id, prot_variants) in enumerate(protein_groups.items()):
        remaining = [v for v in prot_variants if v['rcv_accession'] not in done_rcvs]
        if not remaining:
            continue

        wt_seq = remaining[0]['wt_seq']
        positions = [int(v['protein_position']) - 1 for v in remaining]
        mut_seqs = [v['mut_seq'] for v in remaining]

        spaced_wt = " ".join(wt_seq)
        wt_encoded = tokenizer(spaced_wt, return_tensors="pt")
        wt_input = wt_encoded.input_ids.to(device)
        wt_mask = wt_encoded.attention_mask.to(device)
        with torch.no_grad():
            wt_out = model(input_ids=wt_input, attention_mask=wt_mask)
        wt_embedding = wt_out.last_hidden_state[0].cpu().float()

        max_len = max(len(s) for s in mut_seqs) if mut_seqs else len(wt_seq)
        bs = batch_size_override or get_batch_size(max_len)

        for batch_start in range(0, len(mut_seqs), bs):
            batch_mut_seqs = mut_seqs[batch_start:batch_start + bs]
            batch_positions = positions[batch_start:batch_start + bs]
            batch_variants = remaining[batch_start:batch_start + bs]

            spaced_muts = [" ".join(s) for s in batch_mut_seqs]
            mut_encoded = tokenizer(spaced_muts, padding=True, return_tensors="pt")
            mut_input = mut_encoded.input_ids.to(device)
            mut_mask = mut_encoded.attention_mask.to(device)
            with torch.no_grad():
                mut_out = model(input_ids=mut_input, attention_mask=mut_mask)
            mut_embeddings = mut_out.last_hidden_state.cpu().float()

            wt_vecs, mut_vecs, valid = [], [], []
            for i, (pos, v) in enumerate(zip(batch_positions, batch_variants)):
                tok_pos = pos + 1
                if tok_pos < wt_embedding.shape[0] and tok_pos < mut_embeddings.shape[1]:
                    wt_vecs.append(wt_embedding[tok_pos])
                    mut_vecs.append(mut_embeddings[i, tok_pos])
                    valid.append(v)

            if wt_vecs:
                wt_tensor = torch.stack(wt_vecs)
                mut_tensor = torch.stack(mut_vecs)
                cos_sim = F.cosine_similarity(F.normalize(wt_tensor, dim=1), F.normalize(mut_tensor, dim=1))
                distances = (cos_sim - 1.0).numpy()
                for v, d in zip(valid, distances):
                    results.append({
                        'rcv_accession': v['rcv_accession'],
                        'gene_symbol': v['gene_symbol'],
                        'protein_position': v['protein_position'],
                        'wt_aa': v['wt_aa'], 'mut_aa': v['mut_aa'],
                        'cosine_distance': float(d),
                        'label': v['clinical_significance'],
                    })
                    processed += 1

            done_rcvs.update(v['rcv_accession'] for v in batch_variants)

        if processed % save_interval < bs or prot_idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [protbert] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [protbert] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def run_esm1v(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from embedding_utils import load_esm_model, batch_embed_esm

    print("Loading ESM-1v...")
    model, alphabet = load_esm_model('esm1b')
    model_name_hf = "facebook/esm1v_t33_650M_UR90S_1"
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    hf_model = AutoModelForMaskedLM.from_pretrained(model_name_hf, output_hidden_states=True)
    hf_model = hf_model.to(device).eval()

    for p in hf_model.parameters():
        p.requires_grad = False

    repr_layer = 33
    bs = batch_size_override or get_batch_size(1024)

    protein_groups = group_by_protein(variants)
    checkpoint_path = os.path.join(checkpoint_dir, 'esm1v_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'esm1v_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 500

    for prot_idx, (protein_id, prot_variants) in enumerate(protein_groups.items()):
        remaining = [v for v in prot_variants if v['rcv_accession'] not in done_rcvs]
        if not remaining:
            continue

        wt_seq = remaining[0]['wt_seq']
        positions = [int(v['protein_position']) - 1 for v in remaining]
        mut_seqs = [v['mut_seq'] for v in remaining]

        wt_encoded = tokenizer(wt_seq, return_tensors="pt")
        wt_input = wt_encoded.input_ids.to(device)
        wt_mask = wt_encoded.attention_mask.to(device)
        with torch.no_grad():
            wt_out = hf_model(input_ids=wt_input, attention_mask=wt_mask, output_hidden_states=True)
        wt_embedding = wt_out.hidden_states[repr_layer][0].cpu().float()

        max_len = max(len(s) for s in mut_seqs) if mut_seqs else len(wt_seq)
        bs = batch_size_override or get_batch_size(max_len)

        for batch_start in range(0, len(mut_seqs), bs):
            batch_mut_seqs = mut_seqs[batch_start:batch_start + bs]
            batch_positions = positions[batch_start:batch_start + bs]
            batch_variants = remaining[batch_start:batch_start + bs]

            mut_encoded = tokenizer(batch_mut_seqs, padding=True, return_tensors="pt")
            mut_input = mut_encoded.input_ids.to(device)
            mut_mask = mut_encoded.attention_mask.to(device)
            with torch.no_grad():
                mut_out = hf_model(input_ids=mut_input, attention_mask=mut_mask, output_hidden_states=True)
            mut_embeddings = mut_out.hidden_states[repr_layer].cpu().float()

            wt_vecs, mut_vecs, valid = [], [], []
            for i, (pos, v) in enumerate(zip(batch_positions, batch_variants)):
                tok_pos = pos + 1
                if tok_pos < wt_embedding.shape[0] and tok_pos < mut_embeddings.shape[1]:
                    wt_vecs.append(wt_embedding[tok_pos])
                    mut_vecs.append(mut_embeddings[i, tok_pos])
                    valid.append(v)

            if wt_vecs:
                wt_tensor = torch.stack(wt_vecs)
                mut_tensor = torch.stack(mut_vecs)
                cos_sim = F.cosine_similarity(F.normalize(wt_tensor, dim=1), F.normalize(mut_tensor, dim=1))
                distances = (1.0 - cos_sim).numpy()
                for v, d in zip(valid, distances):
                    results.append({
                        'rcv_accession': v['rcv_accession'],
                        'gene_symbol': v['gene_symbol'],
                        'protein_position': v['protein_position'],
                        'wt_aa': v['wt_aa'], 'mut_aa': v['mut_aa'],
                        'cosine_distance': float(d),
                        'label': v['clinical_significance'],
                    })
                    processed += 1

            done_rcvs.update(v['rcv_accession'] for v in batch_variants)

        if processed % save_interval < bs or prot_idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [esm1v] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [esm1v] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del hf_model
    torch.cuda.empty_cache()
    return results_path


def run_ankh(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "ElnaggarLab/ankh-large"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, output_hidden_states=True)
    encoder = model.encoder.to(device).eval()

    for p in encoder.parameters():
        p.requires_grad = False

    bs = batch_size_override or 4

    protein_groups = group_by_protein(variants)
    checkpoint_path = os.path.join(checkpoint_dir, 'ankh_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'ankh_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 500

    for prot_idx, (protein_id, prot_variants) in enumerate(protein_groups.items()):
        remaining = [v for v in prot_variants if v['rcv_accession'] not in done_rcvs]
        if not remaining:
            continue

        wt_seq = remaining[0]['wt_seq']
        positions = [int(v['protein_position']) - 1 for v in remaining]
        mut_seqs = [v['mut_seq'] for v in remaining]

        wt_encoded = tokenizer(wt_seq, return_tensors="pt", padding=False, truncation=True, max_length=512)
        wt_input = wt_encoded.input_ids.to(device)
        wt_mask = wt_encoded.attention_mask.to(device)
        with torch.no_grad():
            wt_out = encoder(input_ids=wt_input, attention_mask=wt_mask, output_hidden_states=True)
        wt_hidden = wt_out.hidden_states[-1][0].cpu().float()

        for batch_start in range(0, len(mut_seqs), bs):
            batch_mut_seqs = mut_seqs[batch_start:batch_start + bs]
            batch_positions = positions[batch_start:batch_start + bs]
            batch_variants = remaining[batch_start:batch_start + bs]

            mut_encoded = tokenizer(batch_mut_seqs, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt")
            mut_input = mut_encoded.input_ids.to(device)
            mut_mask = mut_encoded.attention_mask.to(device)
            with torch.no_grad():
                mut_out = encoder(input_ids=mut_input, attention_mask=mut_mask, output_hidden_states=True)
            mut_hidden = mut_out.hidden_states[-1].cpu().float()

            wt_vecs, mut_vecs, valid = [], [], []
            for i, (pos, v) in enumerate(zip(batch_positions, batch_variants)):
                tok_pos = pos + 1
                if tok_pos < wt_hidden.shape[0] and tok_pos < mut_hidden.shape[1]:
                    wt_vecs.append(wt_hidden[tok_pos])
                    mut_vecs.append(mut_hidden[i, tok_pos])
                    valid.append(v)

            if wt_vecs:
                wt_tensor = torch.stack(wt_vecs)
                mut_tensor = torch.stack(mut_vecs)
                cos_sim = F.cosine_similarity(F.normalize(wt_tensor, dim=1), F.normalize(mut_tensor, dim=1))
                distances = (1.0 - cos_sim).numpy()
                for v, d in zip(valid, distances):
                    results.append({
                        'rcv_accession': v['rcv_accession'],
                        'gene_symbol': v['gene_symbol'],
                        'protein_position': v['protein_position'],
                        'wt_aa': v['wt_aa'], 'mut_aa': v['mut_aa'],
                        'cosine_distance': float(d),
                        'label': v['clinical_significance'],
                    })
                    processed += 1

            done_rcvs.update(v['rcv_accession'] for v in batch_variants)

        if processed % save_interval < bs or prot_idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [ankh] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [ankh] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del encoder
    torch.cuda.empty_cache()
    return results_path


def run_dnabert1(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    import sys
    import importlib
    from transformers import BertTokenizer, AutoConfig
    from huggingface_hub import snapshot_download

    model_id = "zhihan1996/DNA_bert_6"
    print(f"Loading {model_id}...")

    model_dir = snapshot_download(model_id)
    init_path = os.path.join(model_dir, '__init__.py')
    if not os.path.exists(init_path):
        open(init_path, 'w').close()
    parent = os.path.dirname(model_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    pkg_name = os.path.basename(model_dir)
    mod = importlib.import_module(f'{pkg_name}.dnabert_layer')
    CustomBertModel = mod.BertForMaskedLM

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = CustomBertModel.from_pretrained(model_id, config=config)
    model.config.output_hidden_states = True
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    var_center = 3001
    kmer = 6
    var_token_idx = var_center // kmer
    bs = batch_size_override or 4

    checkpoint_path = os.path.join(checkpoint_dir, 'dnabert1_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'dnabert1_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 200

    remaining = [v for v in variants if v['rcv_accession'] not in done_rcvs]

    for batch_start in range(0, len(remaining), bs):
        batch = remaining[batch_start:batch_start + bs]
        wt_seqs = [v['nuc_context_wt'] for v in batch]
        mut_seqs = [v['nuc_context_mut'] for v in batch]

        def tokenize_kmer(seqs):
            tokenized = []
            for seq in seqs:
                kmers = [seq[i:i+kmer] for i in range(0, len(seq) - kmer + 1, kmer)]
                tokenized.append(" ".join(kmers))
            return tokenized

        wt_tokenized = tokenize_kmer(wt_seqs)
        wt_encoded = tokenizer(wt_tokenized, return_tensors="pt", padding=True, truncation=True, max_length=512)
        wt_input = wt_encoded.input_ids.to(device)
        wt_mask = wt_encoded.attention_mask.to(device)

        mut_tokenized = tokenize_kmer(mut_seqs)
        mut_encoded = tokenizer(mut_tokenized, return_tensors="pt", padding=True, truncation=True, max_length=512)
        mut_input = mut_encoded.input_ids.to(device)
        mut_mask = mut_encoded.attention_mask.to(device)

        with torch.no_grad():
            wt_out = model(input_ids=wt_input, attention_mask=wt_mask, output_hidden_states=True)
            wt_hidden = wt_out.hidden_states[-1].cpu().float()

            mut_out = model(input_ids=mut_input, attention_mask=mut_mask, output_hidden_states=True)
            mut_hidden = mut_out.hidden_states[-1].cpu().float()

        wt_vecs, mut_vecs, valid = [], [], []
        for i, v in enumerate(batch):
            if var_token_idx < wt_hidden.shape[1] and var_token_idx < mut_hidden.shape[1]:
                wt_vecs.append(wt_hidden[i, var_token_idx])
                mut_vecs.append(mut_hidden[i, var_token_idx])
                valid.append(v)

        if wt_vecs:
            wt_tensor = torch.stack(wt_vecs)
            mut_tensor = torch.stack(mut_vecs)
            cos_sim = F.cosine_similarity(F.normalize(wt_tensor, dim=1), F.normalize(mut_tensor, dim=1))
            distances = (1.0 - cos_sim).numpy()
            for v, d in zip(valid, distances):
                results.append({
                    'rcv_accession': v['rcv_accession'],
                    'gene_symbol': v['gene_symbol'],
                    'protein_position': v['protein_position'],
                    'wt_aa': v['wt_aa'], 'mut_aa': v['mut_aa'],
                    'cosine_distance': float(d),
                    'label': v['label'],
                })
                processed += 1

        done_rcvs.update(v['rcv_accession'] for v in batch)

        if processed % save_interval < bs:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [dnabert1] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [dnabert1] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def run_genalm(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    model_id = "AIRI-Institute/gena-lm-bert-base"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, output_hidden_states=True)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    var_center_bp = 3001
    max_tokens = 512
    bs = batch_size_override or 4

    def tokenize_and_find_var(seq):
        encoded = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_tokens,
                            return_offsets_mapping=True)
        offsets = encoded.offset_mapping[0]
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        n_tokens = input_ids.shape[1]

        var_token_idx = 0
        for i, (start, end) in enumerate(offsets):
            if start <= var_center_bp < end:
                var_token_idx = i
                break

        return input_ids, attention_mask, var_token_idx

    checkpoint_path = os.path.join(checkpoint_dir, 'genalm_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'genalm_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 200

    remaining = [v for v in variants if v['rcv_accession'] not in done_rcvs]

    for batch_start in range(0, len(remaining), bs):
        batch = remaining[batch_start:batch_start + bs]
        wt_seqs = [v['nuc_context_wt'] for v in batch]
        mut_seqs = [v['nuc_context_mut'] for v in batch]

        wt_ids_list, wt_masks_list, wt_var_indices = [], [], []
        for seq in wt_seqs:
            ids, mask, idx = tokenize_and_find_var(seq)
            wt_ids_list.append(ids)
            wt_masks_list.append(mask)
            wt_var_indices.append(idx)

        mut_ids_list, mut_masks_list, mut_var_indices = [], [], []
        for seq in mut_seqs:
            ids, mask, idx = tokenize_and_find_var(seq)
            mut_ids_list.append(ids)
            mut_masks_list.append(mask)
            mut_var_indices.append(idx)

        with torch.no_grad():
            wt_embs = []
            for ids, mask in zip(wt_ids_list, wt_masks_list):
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                emb = out.hidden_states[-1][0].cpu().float()
                wt_embs.append(emb)

            mut_embs = []
            for ids, mask in zip(mut_ids_list, mut_masks_list):
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                emb = out.hidden_states[-1][0].cpu().float()
                mut_embs.append(emb)

        wt_vecs, mut_vecs, valid = [], [], []
        for i, v in enumerate(batch):
            widx = wt_var_indices[i]
            midx = mut_var_indices[i]
            if widx < wt_embs[i].shape[0] and midx < mut_embs[i].shape[0]:
                wt_vecs.append(wt_embs[i][widx])
                mut_vecs.append(mut_embs[i][midx])
                valid.append(v)

        if wt_vecs:
            wt_tensor = torch.stack(wt_vecs)
            mut_tensor = torch.stack(mut_vecs)
            cos_sim = F.cosine_similarity(F.normalize(wt_tensor, dim=1), F.normalize(mut_tensor, dim=1))
            distances = (1.0 - cos_sim).numpy()
            for v, d in zip(valid, distances):
                results.append({
                    'rcv_accession': v['rcv_accession'],
                    'gene_symbol': v['gene_symbol'],
                    'protein_position': v['protein_position'],
                    'wt_aa': v['wt_aa'], 'mut_aa': v['mut_aa'],
                    'cosine_distance': float(d),
                    'label': v['label'],
                })
                processed += 1

        done_rcvs.update(v['rcv_accession'] for v in batch)

        if processed % save_interval < bs:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [genalm] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [genalm] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def run_caduceus(variants, device='cuda', checkpoint_dir='results', batch_size_override=None):
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    model_id = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, output_hidden_states=True)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    var_center = 3001
    bs = batch_size_override or 4

    checkpoint_path = os.path.join(checkpoint_dir, 'caduceus_checkpoint.pkl')
    results_path = os.path.join(checkpoint_dir, 'caduceus_results.tsv')

    done_rcvs = set()
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            results = data.get('results', [])
            done_rcvs = set(r['rcv_accession'] for r in results)
        print(f"Resuming from checkpoint: {len(results)} done")

    total = len(variants)
    processed = len(results)
    start_time = time.time()
    save_interval = 200

    remaining = [v for v in variants if v['rcv_accession'] not in done_rcvs]

    for batch_start in range(0, len(remaining), bs):
        batch = remaining[batch_start:batch_start + bs]
        wt_seqs = [v['nuc_context_wt'] for v in batch]
        mut_seqs = [v['nuc_context_mut'] for v in batch]

        wt_encoded = tokenizer(wt_seqs, return_tensors="pt", padding=True, truncation=False)
        wt_input = wt_encoded.input_ids.to(device)
        wt_mask = wt_encoded.attention_mask.to(device)

        mut_encoded = tokenizer(mut_seqs, return_tensors="pt", padding=True, truncation=False)
        mut_input = mut_encoded.input_ids.to(device)
        mut_mask = mut_encoded.attention_mask.to(device)

        with torch.no_grad():
            wt_out = model(input_ids=wt_input, attention_mask=wt_mask, output_hidden_states=True)
            wt_hidden = wt_out.hidden_states[-1].cpu().float()

            mut_out = model(input_ids=mut_input, attention_mask=mut_mask, output_hidden_states=True)
            mut_hidden = mut_out.hidden_states[-1].cpu().float()

        wt_vecs, mut_vecs, valid = [], [], []
        for i, v in enumerate(batch):
            if var_center < wt_hidden.shape[1] and var_center < mut_hidden.shape[1]:
                wt_vecs.append(wt_hidden[i, var_center])
                mut_vecs.append(mut_hidden[i, var_center])
                valid.append(v)

        if wt_vecs:
            wt_tensor = torch.stack(wt_vecs)
            mut_tensor = torch.stack(mut_vecs)
            cos_sim = F.cosine_similarity(F.normalize(wt_tensor, dim=1), F.normalize(mut_tensor, dim=1))
            distances = (1.0 - cos_sim).numpy()
            for v, d in zip(valid, distances):
                results.append({
                    'rcv_accession': v['rcv_accession'],
                    'gene_symbol': v['gene_symbol'],
                    'protein_position': v['protein_position'],
                    'wt_aa': v['wt_aa'], 'mut_aa': v['mut_aa'],
                    'cosine_distance': float(d),
                    'label': v['label'],
                })
                processed += 1

        done_rcvs.update(v['rcv_accession'] for v in batch)

        if processed % save_interval < bs:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [caduceus] {processed}/{total} ({100*processed/total:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results}, f)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rcv_accession', 'gene_symbol', 'protein_position',
            'wt_aa', 'mut_aa', 'cosine_distance', 'label'
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"  [caduceus] Done: {len(results)} variants in {elapsed/60:.1f} min")
    print(f"  Results: {results_path}")

    del model
    torch.cuda.empty_cache()
    return results_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=[
        'esm1b', 'esm2', 'esm3', 'prott5', 'esm1b_mlm', 'esm2_mlm',
        'dnabert2', 'hyenadna', 'ntv2',
        'protbert', 'esm1v', 'ankh', 'dnabert1', 'genalm', 'caduceus',
    ])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--checkpoint_dir', default='results')
    parser.add_argument('--dna_input', default='dna_variants.tsv')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.model in ('dnabert2', 'hyenadna', 'ntv2', 'dnabert1', 'genalm', 'caduceus'):
        print(f"Loading DNA variants from {args.dna_input}...")
        variants = load_variants(args.dna_input)
        print(f"Total DNA variants: {len(variants)}")
        labels = {}
        for v in variants:
            l = v['label']
            labels[l] = labels.get(l, 0) + 1
        print(f"Labels: {labels}")

        if args.model == 'dnabert2':
            run_dnabert2(variants, device=args.device,
                         checkpoint_dir=args.checkpoint_dir,
                         batch_size_override=args.batch_size)
        elif args.model == 'hyenadna':
            run_hyenadna(variants, device=args.device,
                         checkpoint_dir=args.checkpoint_dir,
                         batch_size_override=args.batch_size)
        elif args.model == 'ntv2':
            run_ntv2(variants, device=args.device,
                     checkpoint_dir=args.checkpoint_dir,
                     batch_size_override=args.batch_size)
        elif args.model == 'dnabert1':
            run_dnabert1(variants, device=args.device,
                         checkpoint_dir=args.checkpoint_dir,
                         batch_size_override=args.batch_size)
        elif args.model == 'genalm':
            run_genalm(variants, device=args.device,
                       checkpoint_dir=args.checkpoint_dir,
                       batch_size_override=args.batch_size)
        elif args.model == 'caduceus':
            run_caduceus(variants, device=args.device,
                         checkpoint_dir=args.checkpoint_dir,
                         batch_size_override=args.batch_size)
    else:
        print(f"Loading variants from missense_variants.tsv...")
        variants = load_variants('missense_variants.tsv')
        print(f"Total missense variants: {len(variants)}")
        labels = {}
        for v in variants:
            l = v['clinical_significance']
            labels[l] = labels.get(l, 0) + 1
        print(f"Labels: {labels}")

        if args.model in ('esm1b', 'esm2'):
            run_esm(args.model, variants, device=args.device,
                    checkpoint_dir=args.checkpoint_dir,
                    batch_size_override=args.batch_size)
        elif args.model in ('esm1b_mlm', 'esm2_mlm'):
            base_model = args.model.replace('_mlm', '')
            run_esm_mlm(base_model, variants, device=args.device,
                         checkpoint_dir=args.checkpoint_dir,
                         batch_size_override=args.batch_size)
        elif args.model == 'esm3':
            run_esm3_model(variants, device=args.device,
                           checkpoint_dir=args.checkpoint_dir,
                           batch_size_override=args.batch_size)
        elif args.model == 'prott5':
            run_prott5(variants, device=args.device,
                       checkpoint_dir=args.checkpoint_dir,
                       batch_size_override=args.batch_size)
        elif args.model == 'protbert':
            run_protbert(variants, device=args.device,
                         checkpoint_dir=args.checkpoint_dir,
                         batch_size_override=args.batch_size)
        elif args.model == 'esm1v':
            run_esm1v(variants, device=args.device,
                      checkpoint_dir=args.checkpoint_dir,
                      batch_size_override=args.batch_size)
        elif args.model == 'ankh':
            run_ankh(variants, device=args.device,
                     checkpoint_dir=args.checkpoint_dir,
                     batch_size_override=args.batch_size)


if __name__ == '__main__':
    main()
