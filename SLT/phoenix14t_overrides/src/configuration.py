from transformers import PretrainedConfig


class SLTConfig(PretrainedConfig):
    model_type = "slt_transformer"

    def __init__(
        self,
        input_dim=1024,  # Dimension of your sign features
        d_model=512,  # Transformer hidden size
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        vocab_size=30522,  # Default to BERT vocab size
        max_position_embeddings=1024,
        pad_token_id=0,
        bos_token_id=101,  # BERT CLS
        eos_token_id=102,  # BERT SEP
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
