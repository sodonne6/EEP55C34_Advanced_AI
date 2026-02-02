import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""run one step of training with a fake decoder on I3D features."""

FEATURES = r"SLT\data_test\i3d_features\win_test.npy"

# Toy vocab: pretend we already have SPM ids
# (BOS=1, EOS=2, PAD=0). We'll use a tiny fake target sequence.
PAD, BOS, EOS = 0, 1, 2
target_ids = torch.tensor([[BOS, 10, 11, 12, EOS, PAD, PAD]], dtype=torch.long)  # (B, L)

class TinySLT(nn.Module):
    def __init__(self, d_in=1024, d_model=256, vocab=100):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.out = nn.Linear(d_model, vocab)

    def forward(self, feats, tgt):
        # feats: (B,T,1024) -> memory: (B,T,d_model)
        mem = self.enc(self.proj(feats))

        # teacher forcing: input is tgt[:, :-1], predict tgt[:, 1:]
        inp = tgt[:, :-1]
        gold = tgt[:, 1:]

        x = self.emb(inp)  # (B,L-1,d_model)
        # causal mask for decoder self-attention
        L = x.size(1)
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        y = self.dec(tgt=x, memory=mem, tgt_mask=causal)
        logits = self.out(y)  # (B,L-1,vocab)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               gold.reshape(-1),
                               ignore_index=PAD)
        return loss, logits

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq = np.load(FEATURES).astype(np.float32)  # (T',1024)
    feats = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1,T,1024)
    tgt = target_ids.to(device)

    model = TinySLT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    loss, _ = model(feats, tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print("One training step OK. Loss:", float(loss))

if __name__ == "__main__":
    main()
