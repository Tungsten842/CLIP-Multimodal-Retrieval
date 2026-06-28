import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torchinfo import summary


def collate_fn(batch):
    img_a, img_b, pos_text_embs, neg_text_embs = zip(*batch)

    img_a = torch.stack(img_a)
    img_b = torch.stack(img_b)

    def pad_and_mask(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(max(lengths), 1)
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        if padded.size(1) == 0:
            padded = torch.zeros((len(sequences), 1, sequences[0].shape[-1]))

        mask = torch.arange(max_len)[None, :] >= torch.tensor(lengths)[:, None]

        return padded, mask

    pos_text_embs_padded, pos_mask = pad_and_mask(pos_text_embs)
    neg_text_embs_padded, neg_mask = pad_and_mask(neg_text_embs)

    return img_a, img_b, pos_text_embs_padded, pos_mask, neg_text_embs_padded, neg_mask


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.rms_attn = nn.RMSNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)

        self.rms_mlp = nn.RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, mask):
        x_norm = self.rms_attn(x)
        attn_output = self.self_attn(
            query=x_norm, key=x_norm, value=x_norm, key_padding_mask=mask
        )[0]

        res1 = x + attn_output

        x_mlp = self.rms_mlp(res1)
        x_mlp = self.mlp(x_mlp)

        return res1 + x_mlp


class Model(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.tp_proj = nn.Linear(embed_dim, embed_dim)
        self.tn_proj = nn.Linear(embed_dim, embed_dim)
        self.i_proj = nn.Linear(embed_dim, embed_dim)

        self.layer1 = TransformerLayer(embed_dim=embed_dim, nhead=nhead)
        self.layer2 = TransformerLayer(embed_dim=embed_dim, nhead=nhead)

        summary(self)

    def _prepare_text(self, embeds, mask, proj_layer):
        if embeds is None:
            return None, None

        seq = proj_layer(embeds)
        if mask is None:
            mask = seq.new_zeros(seq.shape[:2], dtype=torch.bool)
        return seq, mask

    def forward(
        self,
        image_embed,
        pos_text_embeds=None,
        neg_text_embeds=None,
        pos_mask=None,
        neg_mask=None,
    ):
        batch_size = image_embed.shape[0]

        text_seqs = []
        text_masks = []

        pos_seq, pos_mask = self._prepare_text(pos_text_embeds, pos_mask, self.tp_proj)
        neg_seq, neg_mask = self._prepare_text(neg_text_embeds, neg_mask, self.tn_proj)

        text_seqs = torch.cat([s for s in (pos_seq, neg_seq) if s is not None], dim=1)
        text_masks = torch.cat(
            [m for m in (pos_mask, neg_mask) if m is not None], dim=1
        )

        image_embed = self.i_proj(image_embed).unsqueeze(1)

        x = torch.cat([image_embed, text_seqs], dim=1)
        img_mask = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=image_embed.device
        )

        mask = torch.cat([img_mask, text_masks], dim=1)

        out = self.layer1(x, mask)
        out = self.layer2(out, mask)
        out = out[:, :1].squeeze(1)

        out = normalize(out, p=2, dim=-1)
        return out
