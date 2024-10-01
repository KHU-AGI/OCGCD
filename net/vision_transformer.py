# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_
# from torch.nn.init import normal_

from net.pet_modules import Adapter, KVLoRA


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.kv_lora = None
    
    def prepare_kv_lora(self, p_leng, prefix, rank=5):
        self.kv_lora = KVLoRA(in_features=self.dim, out_features=self.dim, p_leng=p_leng, prefix=prefix, rank=rank)

    def forward(self, x, prompt=None, prefix=False):
        B, N, C = x.shape
        if self.kv_lora is not None:
            # if self.kv_lora.lora_prefix:
            #     p_leng = self.kv_lora.p_leng
            #     qkv, p_kv = self.kv_lora(module=self.qkv, input= x)
                
            #     p_kv = p_kv.reshape(2*p_leng, 2, 768).permute(1,0,2)
            #     pk = p_kv[0][:p_leng].expand(B,-1,-1).reshape(B, p_leng, self.num_heads, C // self.num_heads).permute(0,2,1,3)
            #     pv = p_kv[1][:p_leng].expand(B,-1,-1).reshape(B, p_leng, self.num_heads, C // self.num_heads).permute(0,2,1,3)
                
            #     qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            #     q, k, v = qkv[0], qkv[1], qkv[2]
                
            #     k = torch.cat([pk,k], dim=-2)
            #     v = torch.cat([pv,v], dim=-2)
                
            # else:
            qkv = self.kv_lora(module=self.qkv, input= x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        
        if prefix:
            #todo prompt: 2, n_length, embed_size
            in_feats = self.qkv.in_features
            k_weight = self.qkv.weight[in_feats:2*in_feats]
            k_bias = self.qkv.bias[in_feats:2*in_feats]
            
            v_weight = self.qkv.weight[2*in_feats:]
            v_bias = self.qkv.bias[2*in_feats:]
            
            pk = F.linear(prompt[1], k_weight, k_bias)   #* n_length, emebed_dim
            k_leng, dim = pk.shape
            pk = pk.expand(B, k_leng, dim).reshape(B, k_leng, self.num_heads, dim//self.num_heads).permute(0,2,1,3)
            
            pv = F.linear(prompt[1], v_weight, v_bias)   #* n_length, emebed_dim
            v_leng, dim = pv.shape
            pv = pv.expand(B, v_leng, dim).reshape(B, v_leng, self.num_heads, dim//self.num_heads).permute(0,2,1,3)

            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
            

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        #todo Adapt MLP Init Position..
        self.a_mlp = None
        self.adapter = None
        

    def prepare_adapt_mlp(self, block_idx, rank):
        print('block indx:{} Adapt MLP inserted'.format(block_idx))
        self.a_mlp = Adapter(down_dim=rank)
        
    def prepare_adapter(self, block_idx):
        print('block indx:{} Adapter inserted'.format(block_idx))
        self.adapter = Adapter(adapt_mlp=False)
    
    # def forward(self, x, prompt=None, prefix=False, return_attention=False):
    #     y, attn = self.attn(self.norm1(x), prompt=prompt, prefix=prefix)
    #     x = x + self.drop_path(y)
        
    #     if self.a_mlp is None:
    #         x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     else:
    #         adapt_x = self.a_mlp(x, add_residual=False)
    #         residual = x
    #         x = self.drop_path(self.mlp(self.norm2(x)))
            
    #         x = x + adapt_x + residual
            
    #     if self.adapter is not None:
    #         x = self.adapter(x)

    #     if return_attention:
    #         return x, attn
    #     else:
    #         return x
    def forward(self, x, prompt=None, prefix=False, return_attention=False):
        y, attn = self.attn(self.norm1(x), prompt=prompt, prefix=prefix)
        x = x + self.drop_path(y)
        
        if self.adapter is not None:
            x = self.adapter(x)
        
        if self.a_mlp is None:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            adapt_x = self.a_mlp(x, add_residual=False)
            residual = x
            x = self.drop_path(self.mlp(self.norm2(x)))
            
            x = x + adapt_x + residual
        

        if return_attention:
            return x, attn
        else:
            return x
    
    def forward_latent(self, x, prompt=None, prefix=False):
        y, _ = self.attn(self.norm1(x), prompt=prompt, prefix=prefix)
        x = x + self.drop_path(y)
        attn_feat = x.clone()
        if self.adapter is not None:
            x = self.adapter(x)
        
        if self.a_mlp is None:
            mlp_feat = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            adapt_x = self.a_mlp(x, add_residual=False)
            residual = x
            mlp_feat = self.drop_path(self.mlp(self.norm2(x)))
            
            x = mlp_feat + adapt_x + residual
            
        

        return x, attn_feat.cpu(), mlp_feat.clone().cpu()


#  def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         if self.config.ffn_adapt and self.config.ffn_option == 'parallel':
#             adapt_x = self.adaptmlp(x, add_residual=False)

#         residual = x
#         x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
#         x = self.drop_path(self.mlp_drop(self.fc2(x)))

#         if self.config.ffn_adapt:
#             if self.config.ffn_option == 'sequential':
#                 x = self.adaptmlp(x)
#             elif self.config.ffn_option == 'parallel':
#                 x = x + adapt_x
#             else:
#                 raise ValueError(self.config.ffn_adapt)

#         x = residual + x
#         return x





class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

#! (embedding_size=args.sz_embedding, pretrained=False, is_norm=args.l2_norm, bn_freeze=args.bn_freeze, num_classes=None)
class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, embedding_size, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        
        self.embedding_size = embedding_size
        
        # self.embedding = nn.Linear(self.embed_dim, self.embedding_size) if self.embedding_size != 0 else nn.Identity()
        self.prefix=False
        self.prompt_tuning_layers=None
        self.a_mlp_layers=None
        self.adapter_layers=None
        self.lora_layers=None
        self.h_prompts=None
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def inc_freeze(self):
        for p in self.parameters():
            p.requires_grad= False
        
        for idx, block in enumerate(self.blocks):
            # if self.nav_layer is not None and idx in self.nav_layer:
            #     for nav_param in block.block_navigator.parameters():
            #         nav_param.requires_grad=True
            
            if self.prompt_tuning_layers is not None:
                self.prompts.requires_grad=True
            
            if self.a_mlp_layers is not None and idx in self.a_mlp_layers:
                for a_mlp_params in block.a_mlp.parameters():
                    a_mlp_params.requires_grad = True
            
            if self.adapter_layers is not None and idx in self.adapter_layers:
                for adapter_params in block.adapter.parameters():
                    adapter_params.requires_grad = True
                
            if self.lora_layers is not None and idx in self.lora_layers:
                for adapter_params in block.attn.kv_lora.parameters():
                    adapter_params.requires_grad = True
    
    def prepare_historical_prompt_tuning(self, n_length):
        print("Prepare Historical Prompt-Tuning..")
        self.h_prompts = nn.Parameter(torch.randn(n_length, self.embedding_size))
        nn.init.normal_(self.h_prompts, 0, 1)
        print("h_prompts:", self.h_prompts.shape)
    
    def prepare_prompt(self, tuning_layers, n_length):
        print("Prepare Prefix Tuning..")
        self.prompt_tuning_layers = tuning_layers
        self.prompts = nn.Parameter(torch.randn(len(tuning_layers), 2, n_length, self.embedding_size))
        nn.init.normal_(self.prompts, 0, 1)
        
        print("Tuning Layer:", self.prompt_tuning_layers)
        print("prompts:", self.prompts.shape)
        self.prefix = True
    
    def prepare_adapt_MLP(self, adapt_mlp_layers, down_dim=5):
        self.a_mlp_layers = adapt_mlp_layers
        for idx, block in enumerate(self.blocks):
            if idx in adapt_mlp_layers:
                block.prepare_adapt_mlp(idx, rank=down_dim)
    
    def prepare_Adapter(self, adapter_layers):
        self.adapter_layers = adapter_layers
        for idx, block in enumerate(self.blocks):
            if idx in adapter_layers:
                block.prepare_adapter(idx)
    
    def prepare_LoRA(self, lora_layers, p_leng=20, prefix=False, down_dim=5):
        self.lora_layers = lora_layers
        for idx, block in enumerate(self.blocks):
            if idx in lora_layers:
                print('block indx:{} LoRA inserted'.format(idx))
                # prepare_kv_lora(self, p_leng, prefix, rank=5)
                block.attn.prepare_kv_lora(p_leng=p_leng, prefix=prefix, rank=down_dim)
        # prepare_kv_lora(self, rank=5)
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)
        # self.pos_embed[:, 0]
        if self.h_prompts is not None:
            h_prompts = self.h_prompts + self.pos_embed[:, 0].expand(self.h_prompts.shape[0],-1)
            h_prompts = h_prompts.expand(x.shape[0], -1, -1)
            # print('x:', x.shape)
            # print('h_prompts:', h_prompts.shape)
            x = torch.cat([x[:, 0].unsqueeze(1), h_prompts, x[:, 1:]], dim=1)
        
        for idx, blk in enumerate(self.blocks):
            if  self.prefix and idx in self.prompt_tuning_layers:
                x = blk(x, prompt=self.prompts[self.prompt_tuning_layers.index(idx)], prefix=self.prefix)
            else:
                x = blk(x)

        x = self.norm(x)
        embed_x = self.embedding(x)
        
        if return_all_patches:
            embed_ori_x = torch.cat([embed_x[:,0].unsqueeze(1), embed_x[:,self.h_prompts.shape[0]+1:]], dim=1)
            prompt_x = embed_x[:,1:self.h_prompts.shape[0]+1]
            return embed_ori_x[:,0], embed_ori_x[:,1:], prompt_x.mean(dim=1)
        else:
            return embed_x[:,0]
        
        # if return_all_patches:
        #     return embed_x
        # else:
        #     return embed_x[:,0]
        
    def save_latent_features(self, x):
        latents = []
        with torch.no_grad():
            x = self.prepare_tokens(x)
            for idx, blk in enumerate(self.blocks):
                if  self.prefix and idx in self.prompt_tuning_layers:
                    # x = blk(x, prompt=self.prompts[self.prompt_tuning_layers.index(idx)], prefix=self.prefix)
                    # forward_latent(self, x, prompt=None, prefix=False)
                    x, attn_feat, mlp_feat = blk.forward_latent(x, prompt=self.prompts[self.prompt_tuning_layers.index(idx)], prefix=self.prefix)
                else:
                    # x = blk(x)
                    x, attn_feat, mlp_feat = blk.forward_latent(x)
                latents.append(attn_feat.cpu())
                latents.append(mlp_feat.cpu())
                
        #* delete CLS token
        # img_latents = [latent[:,1:] for latent in latents]
        # cls_latents = [latent[:,0].unsqueeze(1) for latent in latents]
        return torch.stack([latent[:,1:] for latent in latents]), torch.stack([latent[:,0].unsqueeze(1) for latent in latents])
    

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=65536, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class VisionTransformerWithLinear(nn.Module):

    def __init__(self, base_vit, num_classes=200):

        super().__init__()

        self.base_vit = base_vit
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, return_features=False):

        features = self.base_vit(x)
        features = torch.nn.functional.normalize(features, dim=-1)
        logits = self.fc(features)

        if return_features:
            return logits, features
        else:
            return logits

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.fc.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.fc.weight.copy_(w)
    
class ViT_Dino(nn.Module):
    def __init__(self, args):
        super().__init__()
        # model = VisionTransformer(embedding_size=args.sz_embedding)
        # model.load_state_dict(torch.load(args.vit_pretrained_dino), strict=False)
        self.backbone = VisionTransformer(embedding_size=args.sz_embedding, qkv_bias=True)
        # self.head = DINOHead(in_dim=768, out_dim=65536)
    
    def forward(self, x, return_all_patches=False):
        # forward(self, x, etf_embedding=False, return_all_patches=False)
        return self.backbone(x, return_all_patches)

from timm.models import vit_base_patch16_224

class ViT_IN21K(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = VisionTransformer(embedding_size=args.sz_embedding, qkv_bias=True)
        pretrained_dict = vit_base_patch16_224(pretrained=True).state_dict()
        del pretrained_dict['head.weight']; del pretrained_dict['head.bias']
        self.backbone.load_state_dict(pretrained_dict)
        # self.backbone.head = nn.Identity()
    
    def forward(self, x):
        return self.backbone(x)