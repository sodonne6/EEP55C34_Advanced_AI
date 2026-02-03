import math

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from .configuration import SLTConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, dim]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SLTModel(PreTrainedModel):
    config_class = SLTConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 1. Feature Projection (Sign Features -> d_model)
        self.feature_projection = nn.Linear(config.input_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(
            config.d_model, config.dropout, config.max_position_embeddings
        )
        self.pos_decoder = PositionalEncoding(
            config.d_model, config.dropout, config.max_position_embeddings
        )

        # 2. Vanilla Transformer
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        # 3. Output Head (d_model -> vocab_size)
        self.embed_tgt = nn.Embedding(config.vocab_size, config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

        self.init_weights()
        
    @staticmethod
    def shift_tokens_right(labels, pad_token_id: int, bos_token_id: int):
        # labels: [B, L]
        shifted = labels.new_full(labels.shape, pad_token_id)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = bos_token_id
        return shifted
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_values, attention_mask=None, labels=None, decoder_input_ids=None):
        device = input_values.device

        # Project features
        src = self.feature_projection(input_values)  # [B, S, d_model]
        src = self.pos_encoder(src)

        # Src padding mask for encoder + decoder cross-attn
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # True=pad
        else:
            src_key_padding_mask = None

        # ---- Prepare Decoder Input (FIXED) ----
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Either labels or decoder_input_ids must be provided.")

            # IMPORTANT: replace -100 BEFORE shifting
            labels_for_shift = labels.clone()
            labels_for_shift[labels_for_shift == -100] = self.config.pad_token_id

            decoder_input_ids = self.shift_tokens_right(
                labels_for_shift,
                pad_token_id=self.config.pad_token_id,
                bos_token_id=self.config.bos_token_id,
            )

        decoder_input_ids = decoder_input_ids.long()

        # Decoder padding mask (True=pad)
        tgt_key_padding_mask = (decoder_input_ids == self.config.pad_token_id)

        # Embed + positional encoding
        tgt = self.embed_tgt(decoder_input_ids)
        tgt = self.pos_decoder(tgt)

        # Causal mask
        tgt_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(device)

        # ---- Transformer Pass (FIXED MASKS) ----
        outs = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,  # IMPORTANT
        )

        logits = self.output_head(outs)  # [B, T, vocab]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(loss=loss, logits=logits)
