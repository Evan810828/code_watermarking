"""CodeModel: the auxiliary watermark network of Guo & Cheng (NeurIPS 2025,
"Practical and Effective Code Watermarking for Large Language Models",
../code-watermark). Architecture-only port of src/modelhelper.py.

The network reads the last `context_width` tokens and produces, at the last
position, (a) full-vocab logits distilled from the teacher LLM — their
descending sort defines the rank pairing from which the greenlist is drawn —
and (b) a scalar "switch" logit trained to predict whether the teacher's
next-token entropy exceeds the entropy threshold: the watermark is embedded
and scored only where the switch fires. Detection therefore never needs the
teacher LLM, only this small network.

Port deltas (documented, architecture identical to upstream):
  - compute_loss lives in train_codewm.py, not here (upstream
    modelhelper.py:197 has a bare-text SyntaxError inside compute_loss, so
    the module as shipped does not import; the forward pass is unaffected
    and is ported verbatim).
"""

from typing import Any, Dict

import torch
import torch.nn as nn


class CodeModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        default_config = {
            "max_length": 10,
            "d_model": 512,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 2048,
            "dropout": 0.2,
        }
        if "vocab_size" not in config:
            raise ValueError("vocab_size must be provided in config")
        self.config = default_config.copy()
        self.config.update(config)

        self.wte = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        self.wpe = nn.Embedding(self.config["max_length"], self.config["d_model"])
        self.drop = nn.Dropout(self.config["dropout"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config["num_layers"]
        )
        self.ln_f = nn.LayerNorm(self.config["d_model"])
        # +1: last output unit is the switch logit
        self.output = nn.Linear(self.config["d_model"], self.config["vocab_size"] + 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.kaiming_normal_(
                module.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu"
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_length = input_ids.size()

        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.transformer(src=hidden_states)
        hidden_states = self.ln_f(hidden_states)
        outputs = self.output(hidden_states)

        return {
            "logits": outputs[..., -1, :-1],  # last position, vocab logits
            "switch": outputs[..., -1, -1],  # last position, switch logit
            "full_logits": outputs[..., :-1],
            "full_switch": outputs[..., -1],
        }
