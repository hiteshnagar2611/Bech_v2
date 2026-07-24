import torch
import csv
from transformers import T5EncoderModel, AutoTokenizer

model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
model = model.cuda().eval()

print("Model loaded. Testing on 2 sequences...")

seqs = ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP", "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"]
encoded = tokenizer(seqs, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = encoded.input_ids.cuda()
attention_mask = encoded.attention_mask.cuda()

print(f"Input shape: {input_ids.shape}")

with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask)

embeddings = output.last_hidden_state
print(f"Output shape: {embeddings.shape}")
print(f"Embedding at position 1: {embeddings[0, 1, :5]}")

# Test cosine distance between two different sequences
seq1 = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"
seq2 = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGFFP"  # one aa change at end
encoded = tokenizer([seq1, seq2], padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    out = model(input_ids=encoded.input_ids.cuda(), attention_mask=encoded.attention_mask.cuda())
e1 = out.last_hidden_state[0, 1].float()
e2 = out.last_hidden_state[1, 1].float()
cos_sim = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
print(f"Cosine similarity (pos 1): {cos_sim:.4f}")
print(f"Cosine distance: {1 - cos_sim:.4f}")

print("\nTest passed!")
