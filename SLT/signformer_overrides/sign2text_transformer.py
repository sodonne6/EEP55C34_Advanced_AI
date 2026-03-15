##
## overall issue right now - the majority of points are for the face 
## -the face carries minimal information and the model is holding these points to the same importance as the body and the hands
## - solution: maybe drop the face points and use the 33 pose and 21 hand points only (total 33+21+21=75 points) - this would also reduce the input dimension and make training easier
## - problem: hands sometimes lose tracking so need some way to interpolate between recognised hand frames and non-recognised hand frames so we don't end up with random noise for the hand points when the tracking is lost
##          - possible solution: if hand tracking is lost for a frame (it 0's all hand pooint), take the last recognised hand points and the next recognised hand points and interpolate between them for the frames in between - this way we maintain some kind of hand movement information even when tracking is lost
##  -run ablation:
##  - full model with all points (face+pose+hands)
##  - model with pose and hands only (no face)
##  - add GCN for hands - 3GCN? one for pose, one for left hand, one for right hand - then fuse all three with the transformer branch
##
##

import pdb
import math
import logging
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
)  # fariya

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


class StableSgcnLayer(nn.Module):
    """SGCN block with persistent registered parameters."""

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

        return torch.cat((z1, z2, z3), dim=1)


class StableSgcnLstm(nn.Module):
    """Stable pose SGCN + 3-layer LSTM with registered trainable params."""

    def __init__(self, num_joints: int, ad: torch.Tensor, ad2: torch.Tensor):
        super().__init__()
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
        z = z + y

        b, c, w, t = z.shape
        seq = z.permute(0, 3, 1, 2).contiguous().view(b, t, c * w)

        rec, _ = self.lstm_1(seq)
        rec1, _ = self.lstm_2(rec)
        rec2, _ = self.lstm_3(rec1)
        return rec2


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


@register_model("sign2text_transformer", dataclass=Sign2TextTransformerConfig)
class Sign2TextTransformerModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg, feats_type, feat_dim):
        encoder = Sign2TextTransformerEncoder(cfg, feats_type, feat_dim)
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


class Sign2TextTransformerEncoder(FairseqEncoder):
    """Sign-to-text Transformer encoder + SGCN branch, fused safely back to encoder_embed_dim."""

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

        self.embed_positions = PositionalEmbedding(cfg.max_source_positions, cfg.encoder_embed_dim, self.padding_idx)

        self.transformer_layers = nn.ModuleList([TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers)])
        self.layer_norm = LayerNorm(cfg.encoder_embed_dim) if cfg.encoder_normalize_before else None

        # sgcn branch with persistent registered params
        pose_graph = Graph(33)
        self.encoder2 = StableSgcnLstm(33, pose_graph.AD, pose_graph.AD2)

        # ✅ NEW: fusion projection (concat 256+256=512 -> back to 256)
        self.fuse_proj = nn.Linear(cfg.encoder_embed_dim * 2, cfg.encoder_embed_dim)

    def forward(self, src_tokens, src_mediapipe_tokens, encoder_padding_mask, mediapipe_padding_mask, return_all_hiddens=False):
        # -------- main branch --------
        if self.feats_type == SignFeatsType.mediapipe:
            src_tokens = src_tokens.view(src_tokens.shape[0], src_tokens.shape[1], -1)  # (B,T,1629)

        x = self.feat_proj(src_tokens).transpose(0, 1)  # (T,B,C=256)
        x = self.embed_scale * x

        # positions: feed tokens shaped (B,T), pad tokens = padding_idx
        #pos_input = encoder_padding_mask.long() * self.padding_idx
        #positions = self.embed_positions(pos_input).transpose(0, 1)  # (T,B,256)
        
        #shane - fix: padding_idx gives you a tensor of all 1/0 so most timesteps get the same pos embedding
        # instead create a dummy token tensor shape B T where pads are padding_idx and non pads are anything else (usually 0)
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

        # -------- sgcn branch (fix shapes for AD=33) --------
        # src_mediapipe_tokens: (B,T,543,3)
        mp = src_mediapipe_tokens

        # pose-only landmarks: 33 points - shane - changed so it aligns with the setup from the mediapipe npy file format we are using - this could explain why training was a bit stunted before 
        mp = mp[:, :, 0:33, :]  # (B,T,33,3)

        # sgcn expects (N,C,W,T)
        # shane - currently the GCN only uses the 33 pose landmarks - later we can try to add hands GCN
        mp = mp.permute(0, 3, 2, 1).contiguous()  # (B,3,33,T)

        if self.num_updates < 1:
            print("DEBUG src_tokens(main):", src_tokens.shape, src_tokens.dtype, src_tokens.device)
            print("DEBUG mp(for sgcn):", mp.shape, mp.dtype, mp.device)

        # ensure sgcn module on same device
        #self.encoder2 = self.encoder2.to(mp.device)
        #self.fuse_proj = self.fuse_proj.to(mp.device)

        x2 = self.encoder2(mp)

        # normalize sgcn output to (T,B,C2)
        if x2.dim() == 3:
            # assume (B,T,C2)
            x2 = x2.permute(1, 0, 2).contiguous()
        elif x2.dim() == 2:
            x2 = x2.unsqueeze(0)  # (1,B,C2)
        else:
            raise RuntimeError(f"Sgcn_Lstm returned unexpected shape: {tuple(x2.shape)}")

        # -------- fuse (concat then project back to encoder_embed_dim) --------
        # ensure time alignment
        if x2.size(0) != x.size(0):
            T = max(x.size(0), x2.size(0))
            if x.size(0) < T:
                pad = torch.zeros(T - x.size(0), x.size(1), x.size(2), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            if x2.size(0) < T:
                pad = torch.zeros(T - x2.size(0), x2.size(1), x2.size(2), device=x2.device, dtype=x2.dtype)
                x2 = torch.cat([x2, pad], dim=0)

        fused = torch.cat([x, x2], dim=2)           # (T,B,512)
        fused_output = self.fuse_proj(fused)        # ✅ (T,B,256) so decoder encoder_attn works

        # IMPORTANT: keep padding mask aligned to time dimension
        encoder_padding_mask_combined = encoder_padding_mask

        return {
            "encoder_out": [fused_output],  # T x B x C(256)
            "encoder_padding_mask": [encoder_padding_mask_combined] if encoder_padding_mask_combined.any() else [],
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
