# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    #* Parallel Adapter: Low rank: 5
    def __init__(self, embed_dim=768, down_dim=64, adapt_mlp=True):
        super().__init__()
        self.n_embd = embed_dim
        self.adapt_mlp = adapt_mlp
        
        self.down_size = down_dim
        self.non_linear_func = nn.ReLU()
        if not self.adapt_mlp:
            self.down_size = 10
            self.non_linear_func = nn.GELU()
        

        self.adapter_layer_norm_before = None
        self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        self.scale = nn.Parameter(torch.tensor(0.1))
        
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        
        self.up_proj = nn.Linear(self.down_size, self.n_embd)


    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        # if self.adapter_layernorm_option == 'in':
        x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        if self.adapt_mlp:
            up = up * self.scale

        # if self.adapter_layernorm_option == 'out':
            # up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
    

from typing import Optional, Tuple, Union
import torch.nn.functional as F
class Scaler(nn.Module):
    def __init__(self, scale: Optional[float] = None):
        super().__init__()

        if scale is None:
            self.register_parameter("scale", nn.Parameter(torch.tensor(1.0)))
        else:
            self.scale = scale

    def forward(self, input):
        return input * self.scale

class KVLoRA(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        p_leng=20,
        prefix=False,
        rank: Union[int, Tuple[int]] = 5,
        scale: Union[None, float, Tuple[float, float]] = None,
    ):
        super().__init__()

        assert rank > 0

        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.zeros((rank, in_features))) for _ in range(2)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros((out_features, rank))) for _ in range(2)]
        )
        # self.lora_prefix = prefix
        # if self.lora_prefix:
            # self.p_leng = p_leng
            # self.prompt = nn.Parameter(torch.randn(2*p_leng, in_features))
            # self.p_act = nn.GELU()
        
        if not isinstance(scale, tuple):
            scale = (scale, scale)
        self.scale = nn.ModuleList([Scaler(scale[0]), Scaler(scale[1])])

        self.reset_parameters()
        
    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        for i in range(2):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i])
    
    def forward(self, module: nn.Linear, input: torch.Tensor):
        #todo module --> self.qkv
        items = zip(self.scale, self.lora_A, self.lora_B)
        weight = torch.cat([s(B @ A) for s, A, B in items], dim=0)
        zeros = torch.zeros_like(module.weight)[: -weight.shape[0]]
        weight = torch.cat([zeros, weight], dim=0)
        
        output = F.linear(input, module.weight + weight, module.bias)
        # if self.lora_prefix:
            # p_output = F.linear(self.prompt, module.weight[zeros.shape[0]:] + weight[zeros.shape[0]], module.bias[zeros.shape[0]])
            # return output, self.p_act(p_output)
        # else:   #* Naive LoRA
        return output