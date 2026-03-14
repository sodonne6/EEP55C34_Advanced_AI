##
## 3-GCN variant of sign2text_transformer.py
##
## Architecture:
##   - Main branch: i3d (1024-dim) or mediapipe holistic (543-dim) through Transformer encoder
##   - SGCN branch 1 (pose):       33 MediaPipe pose landmarks  → Sgcn_Lstm  (33-node pose graph)
##   - SGCN branch 2 (left hand):  21 MediaPipe hand landmarks  → Sgcn_Lstm_Hand (21-node hand graph)
##   - SGCN branch 3 (right hand): 21 MediaPipe hand landmarks  → Sgcn_Lstm_Hand (21-node hand graph)
##   Fusion: cat(main, pose_gcn, lh_gcn, rh_gcn) → Linear(dim*4, dim)
##
## Dataset note:
##   For feats_type==i3d the mediapipe .npy file must contain 75 landmarks (T, 75, 3):
##     indices 0:33  → pose landmarks
##     indices 33:54 → left-hand landmarks
##     indices 54:75 → right-hand landmarks
##   (Previously only 33 pose landmarks were required.)
##

import math
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from omegaconf import II

import torch
import torch.nn as nn
from torch import Tensor

from pose_format import Pose

from fairseq import checkpoint_utils, utils

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.data.sign_language import SignFeatsType

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum

from fairseq.models.sign_to_text.graph import Graph
from fairseq.models.sign_to_text.dim_reduction import (
    DimensionReductionLayerLinear,
    DimensionReductionLayerLSTM,
)

from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
)

from fairseq.models.transformer import Embedding, TransformerDecoder

from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hand skeleton graph (21-node MediaPipe hand landmarks)
# ---------------------------------------------------------------------------

class HandGraph:
    """
    21-node graph for MediaPipe hand landmarks.

    Landmark index layout (same for left and right hand):
        0:  wrist
        1-4:  thumb  (MCP → IP → TIP)
        5-8:  index  (MCP → PIP → DIP → TIP)
        9-12: middle
        13-16: ring
        17-20: pinky
    """

    def __init__(self):
        self.num_node = 21
        self.AD, self.AD2, self.bias_mat_1, self.bias_mat_2 = self._build()

    def _build(self):
        N = self.num_node
        self_link = [(i, i) for i in range(N)]
        neighbor_link = [
            # thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # palm cross-connections
            (5, 9), (9, 13), (13, 17),
        ]
        edge = self_link + neighbor_link

        A = np.zeros((N, N), dtype=np.float32)
        for i, j in edge:
            A[i, j] = 1.0
            A[j, i] = 1.0

        A2 = np.zeros((N, N), dtype=np.float32)
        for root in range(N):
            for nb in range(N):
                if A[root, nb]:
                    for nb2 in range(N):
                        if A[nb, nb2]:
                            A2[root, nb2] = 1.0

        bias1 = np.where(A != 0, 0.0, -1e9).astype(np.float32)
        bias2 = np.where(A2 != 0, A2, -1e9).astype(np.float32)

        return (
            torch.tensor(A),
            torch.tensor(A2),
            torch.tensor(bias1),
            torch.tensor(bias2),
        )


# ---------------------------------------------------------------------------
# Sgcn_Lstm variant for 21-node hand graph
# ---------------------------------------------------------------------------

class StableSgcnLayer(nn.Module):
    """A stable (registered-parameter) SGCN block matching the original math."""

    def __init__(self, ad: torch.Tensor, ad2: torch.Tensor, in_channels: int):
        super().__init__()
        self.register_buffer("ad", ad.float())
        self.register_buffer("ad2", ad2.float())

        self.conv_t = nn.Conv2d(in_channels, 64, kernel_size=(9, 1), padding=(4, 0))
        self.conv_x = nn.Conv2d(in_channels + 64, 64, kernel_size=(1, 1))
        self.conv_y = nn.Conv2d(in_channels + 64, 64, kernel_size=(1, 1))

        self.conv_z1 = nn.Conv2d(128, 16, kernel_size=(9, 1), padding=(4, 0))
        self.conv_z2 = nn.Conv2d(16, 16, kernel_size=(15, 1), padding=(7, 0))
        self.conv_z3 = nn.Conv2d(16, 16, kernel_size=(19, 1), padding=(9, 0))
        self.dropout = nn.Dropout2d(p=0.25)

    def forward(self, x):
        # x: (B, C, W, T)
        k1 = torch.relu(self.conv_t(x))
        k = torch.cat((x, k1), dim=1)

        x1 = torch.relu(self.conv_x(k))
        gcn_x1 = torch.einsum("vw,ncwt->ncvt", self.ad.to(x.device), x1)

        y1 = torch.relu(self.conv_y(k))
        gcn_y1 = torch.einsum("vw,ncwt->ncvt", self.ad2.to(x.device), y1)

        gcn_1 = torch.cat((gcn_x1, gcn_y1), dim=1)

        z1 = self.dropout(torch.relu(self.conv_z1(gcn_1)))
        z2 = self.dropout(torch.relu(self.conv_z2(z1)))
        z3 = self.dropout(torch.relu(self.conv_z3(z2)))

        return torch.cat((z1, z2, z3), dim=1)  # (B, 48, W, T)


class StableSgcnLstm(nn.Module):
    """
    SGCN + 3-layer LSTM with persistent registered parameters.

    Input:  (B, 3, W, T)
    Output: (B, T, 256)
    """

    def __init__(self, num_joints: int, ad: torch.Tensor, ad2: torch.Tensor):
        super().__init__()
        self.num_joints = num_joints
        self.output_dim = 256

        self.sgcn_1 = StableSgcnLayer(ad, ad2, in_channels=3)
        self.sgcn_2 = StableSgcnLayer(ad, ad2, in_channels=48)
        self.sgcn_3 = StableSgcnLayer(ad, ad2, in_channels=48)

        self.lstm_1 = nn.LSTM(
            input_size=48 * num_joints,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.lstm_2 = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.lstm_3 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

    def forward(self, x):
        # x: (B, 3, W, T)
        x = self.sgcn_1(x)
        y = self.sgcn_2(x)
        y = y + x
        z = self.sgcn_3(y)
        z = z + y  # (B, 48, W, T)

        b, c, w, t = z.shape
        seq = z.permute(0, 3, 1, 2).contiguous().view(b, t, c * w)

        rec, _ = self.lstm_1(seq)
        rec1, _ = self.lstm_2(rec)
        rec2, _ = self.lstm_3(rec1)
        return rec2


# ---------------------------------------------------------------------------
# Config (identical to base model)
# ---------------------------------------------------------------------------

@dataclass
class Sign2TextTransformerConfig(FairseqDataclass):
    """Add model-specific arguments to the parser."""
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability after activation in FFN."}
    )

    encoder_embed_dim: int = field(default=512, metadata={"help": "encoder embedding dimension"})
    encoder_ffn_embed_dim: int = field(default=2048, metadata={"help": "encoder embedding dimension for FFN"})
    encoder_layers: int = field(default=12, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(default=8, metadata={"help": "num encoder attention heads"})
    encoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each encoder block"}
    )

    decoder_embed_dim: int = field(default=512, metadata={"help": "decoder embedding dimension"})
    decoder_ffn_embed_dim: int = field(default=2048, metadata={"help": "decoder embedding dimension for FFN"})
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(default=8, metadata={"help": "num decoder attention heads"})
    decoder_output_dim: int = field(
        default=512,
        metadata={"help": "decoder output dimension (extra linear layer if different from decoder embed dim)"},
    )
    decoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each decoder block"}
    )

    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    layernorm_embedding: bool = field(default=False, metadata={"help": "add layernorm to embedding"})
    no_scale_embedding: bool = field(default=False, metadata={"help": "if True, dont scale embeddings"})

    load_pretrained_encoder_from: Optional[str] = field(
        default=None, metadata={"help": "model to take encoder weights from (for initialization)"}
    )
    load_pretrained_decoder_from: Optional[str] = field(
        default=None, metadata={"help": "model to take decoder weights from (for initialization)"}
    )

    max_source_positions: int = II("task.max_source_positions")
    max_target_positions: int = II("task.max_target_positions")
    feats_type: ChoiceEnum([x.name for x in SignFeatsType]) = II("task.feats_type")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@register_model("sign2text_transformer_3gcn", dataclass=Sign2TextTransformerConfig)
class Sign2TextTransformerModel3GCN(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg, feats_type, feat_dim):
        encoder = Sign2TextTransformerEncoder3GCN(cfg, feats_type, feat_dim)
        pretraining_path = getattr(cfg, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(f"skipped pretraining because {pretraining_path} does not exist")
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, cfg, task, embed_tokens):
        decoder = TransformerDecoder(cfg, task.target_dictionary, embed_tokens)
        pretraining_path = getattr(cfg, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(f"skipped pretraining because {pretraining_path} does not exist")
            else:
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained decoder from: {pretraining_path}")
        return decoder

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        if cfg.feats_type == SignFeatsType.i3d:
            feat_dim = 1024
        elif cfg.feats_type == SignFeatsType.mediapipe:
            feat_dim = 543

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(task.target_dictionary, cfg.decoder_embed_dim)
        encoder = cls.build_encoder(cfg, cfg.feats_type, feat_dim)
        decoder = cls.build_decoder(cfg, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_mediapipe_tokens, encoder_padding_mask, mediapipe_padding_mask, prev_output_tokens):
        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_mediapipe_tokens=src_mediapipe_tokens,
            encoder_padding_mask=encoder_padding_mask,
            mediapipe_padding_mask=mediapipe_padding_mask,
        )
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)
        return decoder_out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Sign2TextTransformerEncoder3GCN(FairseqEncoder):
    """
        Sign-to-text Transformer encoder with three parallel SGCN branches:
            - encoder2    : 33-node pose GCN      (StableSgcnLstm)
            - encoder2_lh : 21-node left-hand GCN (StableSgcnLstm)
            - encoder2_rh : 21-node right-hand GCN (StableSgcnLstm)

    All three GCN outputs (each (B,T,256)) plus the main Transformer branch
    are concatenated along the feature axis and projected back to encoder_embed_dim.

    Mediapipe input (src_mediapipe_tokens) must have shape (B, T, 75, 3):
        [:, :,  0:33, :] → pose
        [:, :, 33:54, :] → left hand
        [:, :, 54:75, :] → right hand
    """

    def __init__(self, cfg, feats_type: SignFeatsType, feat_dim: int):
        super().__init__(None)

        self.num_updates = 0
        self.dropout_module = FairseqDropout(p=cfg.dropout, module_name=self.__class__.__name__)
        self.embed_scale = math.sqrt(cfg.encoder_embed_dim)
        if cfg.no_scale_embedding:
            self.embed_scale = 1.0

        self.padding_idx = 1
        self.feats_type = feats_type
        self.encoder_embed_dim = cfg.encoder_embed_dim

        # main input projection
        if feats_type in (SignFeatsType.mediapipe, SignFeatsType.openpose):
            self.feat_proj = nn.Linear(feat_dim * 3, cfg.encoder_embed_dim)
        elif feats_type == SignFeatsType.i3d:
            self.feat_proj = nn.Linear(feat_dim, cfg.encoder_embed_dim)

        self.embed_positions = PositionalEmbedding(
            cfg.max_source_positions, cfg.encoder_embed_dim, self.padding_idx
        )
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        )
        self.layer_norm = LayerNorm(cfg.encoder_embed_dim) if cfg.encoder_normalize_before else None

        # ── three SGCN branches ──────────────────────────────────────────────
        pose_graph = Graph(33)
        hand_graph = HandGraph()

        self.encoder2 = StableSgcnLstm(33, pose_graph.AD, pose_graph.AD2)
        self.encoder2_lh = StableSgcnLstm(21, hand_graph.AD, hand_graph.AD2)
        self.encoder2_rh = StableSgcnLstm(21, hand_graph.AD, hand_graph.AD2)

        # ── fusion projection: (main_dim + 3*gcn_dim) → main_dim ────────────
        self.gcn_out_dim = self.encoder2.output_dim
        fuse_in_dim = cfg.encoder_embed_dim + 3 * self.gcn_out_dim
        self.fuse_proj = nn.Linear(fuse_in_dim, cfg.encoder_embed_dim)

    # ------------------------------------------------------------------
    # Helper: normalise a GCN output to (T, B, C)
    # ------------------------------------------------------------------
    @staticmethod
    def _gcn_to_tbc(x2):
        if x2.dim() == 3:
            return x2.permute(1, 0, 2).contiguous()   # (B, T, C) → (T, B, C)
        elif x2.dim() == 2:
            return x2.unsqueeze(0)                     # (B, C) → (1, B, C)
        raise RuntimeError(f"Unexpected GCN output shape: {tuple(x2.shape)}")

    # ------------------------------------------------------------------
    # Helper: pad a (T2, B, C) tensor to length T along dim 0
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_to(tensor, T):
        t = tensor.size(0)
        if t == T:
            return tensor
        pad = torch.zeros(T - t, tensor.size(1), tensor.size(2),
                          device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=0)

    def forward(
        self,
        src_tokens,
        src_mediapipe_tokens,
        encoder_padding_mask,
        mediapipe_padding_mask,
        return_all_hiddens=False,
    ):
        # ── main Transformer branch ───────────────────────────────────────────
        if self.feats_type == SignFeatsType.mediapipe:
            src_tokens = src_tokens.view(src_tokens.shape[0], src_tokens.shape[1], -1)

        x = self.feat_proj(src_tokens).transpose(0, 1)   # (T, B, dim)
        x = self.embed_scale * x

        pos_input = torch.zeros_like(encoder_padding_mask, dtype=torch.long)
        pos_input = pos_input.masked_fill(encoder_padding_mask, self.padding_idx)
        positions = self.embed_positions(pos_input).transpose(0, 1)

        x = x + positions
        x = self.dropout_module(x)

        encoder_states = []
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # ── extract pose / left-hand / right-hand from mediapipe tokens ───────
        # src_mediapipe_tokens expected: (B, T, 75, 3)
        #   [0:33]  pose lankmarks
        #   [33:54] left-hand landmarks
        #   [54:75] right-hand landmarks
        mp = src_mediapipe_tokens                     # (B, T, 75, 3)

        pose_mp = mp[:, :,  0:33, :].permute(0, 3, 2, 1).contiguous()   # (B, 3, 33, T)
        lh_mp   = mp[:, :, 33:54, :].permute(0, 3, 2, 1).contiguous()   # (B, 3, 21, T)
        rh_mp   = mp[:, :, 54:75, :].permute(0, 3, 2, 1).contiguous()   # (B, 3, 21, T)

        if self.num_updates < 1:
            print("DEBUG src_tokens(main):", src_tokens.shape, src_tokens.dtype, src_tokens.device)
            print("DEBUG pose_mp:", pose_mp.shape, "  lh:", lh_mp.shape, "  rh:", rh_mp.shape)

        # ── run the three GCNs ────────────────────────────────────────────────
        x2 = self.encoder2(pose_mp)       # (B, T, 256)
        x3 = self.encoder2_lh(lh_mp)      # (B, T, 256)
        x4 = self.encoder2_rh(rh_mp)      # (B, T, 256)

        x2 = self._gcn_to_tbc(x2)         # (T, B, 256)
        x3 = self._gcn_to_tbc(x3)         # (T, B, 256)
        x4 = self._gcn_to_tbc(x4)         # (T, B, 256)

        # ── align all branches to the same temporal length ────────────────────
        T = max(x.size(0), x2.size(0), x3.size(0), x4.size(0))
        x  = self._pad_to(x,  T)
        x2 = self._pad_to(x2, T)
        x3 = self._pad_to(x3, T)
        x4 = self._pad_to(x4, T)

        # ── fuse: cat along feature dim then project back to encoder_embed_dim ─
        fused = torch.cat([x, x2, x3, x4], dim=2)   # (T, B, encoder_dim + 3*256)
        fused_output = self.fuse_proj(fused)         # (T, B, encoder_dim)

        return {
            "encoder_out": [fused_output],
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],
            "encoder_embedding": [],
            "encoder_states": encoder_states,
            "src_tokens": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )
        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )
        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": new_encoder_embedding,
            "encoder_states": encoder_states,
            "src_tokens": [],
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
