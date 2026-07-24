import torch
print("Loading ESM3-open...")
from esm.models.esm3 import ESM3
model = ESM3.from_pretrained("esm3-open", device=torch.device("cpu"))
model = model.cuda().float()
model = model.eval()
print("Model ready (float32 on CUDA)")

from esm.sdk.api import ESMProtein

seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV"

with torch.no_grad():
    protein = ESMProtein(sequence=seq)
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
            forward_input[dst] = val

    print("Forward inputs:", {k: v.shape for k, v in forward_input.items()})

    output = model.forward(**forward_input)
    print("Output embeddings shape:", output.embeddings.shape)

    pos = 1
    embed = output.embeddings[0, pos]
    print(f"Embedding at pos {pos}: shape={embed.shape}, dtype={embed.dtype}")
    print("Sample values:", embed[:5])
    print("\nSUCCESS!")
