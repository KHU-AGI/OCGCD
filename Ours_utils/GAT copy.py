import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#todo GAT Module for Prediction and Clustering (Main Module!)
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.leakyReLU = nn.LeakyReLU(0.2)
        # self.leakyReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        #* q,k,v -> H, B, H_dim(dim//Heads)
        attn = torch.bmm(q, k.transpose(1, 2))  #* H, B, B  #* Feature Attention (Including Head node)
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, -1)
        attn_ = self.softmax(self.leakyReLU(attn))
        attn = self.dropout(attn_)
        output = torch.bmm(attn, v)     #*  H, B, H_dim
        return output, attn_, log_attn


#* hdim=self.num_features
#* self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

class GAT_layer(nn.Module):
    ''' Multi-Head Attention module '''

    # def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    def __init__(self, n_head, d_model, d_feat, dropout=0.1):
        super(GAT_layer, self).__init__()
        self.n_head = n_head    #* 12
        # self.d_k = d_k          #* 768
        # self.d_v = d_v          #* 768 = 12 * 64
        self.d_feat = d_feat

        self.w_qs = nn.Linear(d_model, d_feat, bias=False)
        self.w_ks = nn.Linear(d_model, d_feat, bias=False)
        self.w_vs = nn.Linear(d_model, d_feat, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_feat)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_feat)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_feat)))
        
        self.attention = ScaledDotProductAttention(temperature=1.)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_feat, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, in_feats, return_attn=False, skip_connect=True):
    def forward(self, in_feats, in_heads, last_layer=False, skip_connect=True):
        feat_n = in_feats.shape[0]
        B = in_feats.shape[0] + in_heads.shape[0]
        dim = in_feats.shape[-1]
        
        x = torch.cat([in_feats, in_heads], dim=0)
        residual = x
        
        in_feats = self.layer_norm(in_feats)
        q = self.w_qs(x) #* B, dim
        k = self.w_ks(x) #* B, dim
        v = self.w_vs(x) #* B, dim
        
        # q = torch.cat([fq, hq], dim=0)
        # k = torch.cat([fk, hk], dim=0)
        # v = torch.cat([fv, hv], dim=0)

        #* B, dim --> H, B(=N), dim/H 
        q = q.reshape(B, self.n_head, dim//self.n_head).permute(1,0,2)
        k = k.reshape(B, self.n_head, dim//self.n_head).permute(1,0,2)
        v = v.reshape(B, self.n_head, dim//self.n_head).permute(1,0,2)
        
        output, attn, log_attn = self.attention(q, k, v)

        output = output.permute(1, 0, 2).reshape(B, dim)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        # output = self.layer_norm(output + residual)
        if skip_connect:
            output += residual

        if last_layer:
            return output
        else:
            return output[:feat_n], output[feat_n:]
    
    def forward_prefix(self, inputs, prefix):
        #todo inputs (Heads): query, key, value
        #todo prefix (Embedding features): key, value
        B, dim = inputs.shape
        N, _ = prefix.shape
        
        residual = inputs
        
        inputs = self.layer_norm(inputs)
        q = self.w_qs(inputs) #* B, dim
        inp_k = self.w_ks(inputs) #* B, dim
        inp_v = self.w_vs(inputs) #* B, dim
        
        prefix = self.layer_norm(prefix)
        prefix_k = self.w_ks(prefix) #* N, dim
        prefix_v = self.w_vs(prefix) #* N, dim
        
        k = torch.cat([prefix_k, inp_k], dim=0) #* B+N, dim
        v = torch.cat([prefix_v, inp_v], dim=0) #* B+N, dim

        #* B, dim --> H, B(=N), dim/H 
        q = q.reshape(B, self.n_head, dim//self.n_head).permute(1,0,2)
        k = k.reshape(B+N, self.n_head, dim//self.n_head).permute(1,0,2)
        v = v.reshape(B+N, self.n_head, dim//self.n_head).permute(1,0,2)
        
        # output, attn, log_attn = self.attention(q, k, v)
        attn = torch.bmm(q, k.transpose(1, 2))  #* H, B, B  #* Feature Attention (Including Head node)
        attn = attn / self.attention.temperature
        attn_ = self.attention.softmax(self.attention.leakyReLU(attn))
        attn = self.attention.dropout(attn_)
        output = torch.bmm(attn, v)     #*  H, B, H_dim
        
        output = output.permute(1, 0, 2).reshape(B, dim)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output += residual
        
        return output


class GAT_module(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_feat, dropout=0.1, num_layers=4):
        super(GAT_module, self).__init__()
        #todo Layer 갯수 만큼 
        self.gat_layers = nn.Sequential(*[GAT_layer(n_head=n_head, d_model=d_model, d_feat=d_feat) for _ in range(num_layers)])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, input_feats, in_heads, return_attn=False):
        # input_feats = self.gat_layers(input_feats)
        for l_idx, gat_layer in enumerate(self.gat_layers):
            if (l_idx+1) != len(self.gat_layers):
                input_feats, in_heads = gat_layer(input_feats, in_heads, skip_connect=False)
            else:
                outputs = gat_layer(input_feats, in_heads, last_layer=True)
            # input_feats = gat_layer(input_feats, in_heads)
        # return input_feats
        return outputs
    
    # def prefix_tuning(self, heads, prefix):
    #     for gat_layer in self.gat_layers:
    #         heads = gat_layer.forward_prefix(heads, prefix)
        
    #     return heads
    
    # def forward_prefix(self, inputs, prefix):
    #     #todo inputs: query, key, value
    #     #todo prefix: key, value
    #     q = gat_layer.w_qs(gat_feats) #* B, dim
    #     k = gat_layer.w_ks(gat_combined) #* B+Heads, dim
    #     v = gat_layer.w_vs(in_feats) #* B, dim
        
    #     q = q.reshape(B, self.n_head, dim//self.n_head).permute(1,0,2)
    #     k = k.reshape(B, self.n_head, dim//self.n_head).permute(1,0,2)
    #     v = v.reshape(B, self.n_head, dim//self.n_head).permute(1,0,2)
        
    #     attn = torch.bmm(q, k.transpose(1, 2))  #* H, B, B  #* Feature Attention (Including Head node)
    #     attn = attn / self.temperature
    #     log_attn = F.log_softmax(attn, -1)
    #     attn_ = self.softmax(self.leakyReLU(attn))
    #     attn = self.dropout(attn_)
    #     output = torch.bmm(attn, v)     #*  H, B, H_dim
        
    #     pass
    
    def get_attn_score(self, in_feats, in_heads):
        # head_attns_list=[]
        # feats_attns_list=[]
        attn_list = []
        
        for gat_layer in self.gat_layers:
            attn_dicts={}
            
            gat_feats = gat_layer.layer_norm(in_feats)
            gat_heads = gat_layer.layer_norm(in_heads)
            
            feat_N, dim = gat_feats.shape
            head_N, dim = gat_heads.shape
            # q = gat_layer.w_qs(gat_feats) #* B, dim
            # q = gat_layer.w_qs(gat_combined) #* B, dim
            # k = gat_layer.w_ks(gat_combined) #* B+Heads, dim
            fq = gat_layer.w_qs(in_feats) #* B, dim
            fk = gat_layer.w_ks(in_feats) #* B, dim
            
            hq = gat_layer.w_qs(in_heads) #* B, dim
            hk = gat_layer.w_ks(in_heads) #* B, dim
            
            q = torch.cat([fq, hq], dim=0)
            k = torch.cat([fk, hk], dim=0)
            
            # feat_B, dim = gat_feats.shape
            # combined_B, _ = q.shape
            
            q = q.reshape(feat_N + head_N, gat_layer.n_head, dim//gat_layer.n_head).permute(1,0,2)
            k = k.reshape(feat_N + head_N, gat_layer.n_head, dim//gat_layer.n_head).permute(1,0,2)
            
        
            # v = gat_layer.w_vs(in_feats) #* B, dim
            attn = torch.bmm(q, k.transpose(1, 2))  #* H, B, B+Heads  #* Feature Attention (Including Head node)
            attn = attn / gat_layer.attention.temperature
            attn = gat_layer.attention.softmax(gat_layer.attention.leakyReLU(attn)).mean(dim=0) #* B x (B+Heads)
            feats_feats_attn = attn[:feat_N, :feat_N]
            feats_heads_attn = attn[:feat_N, feat_N:]
            heads_feats_attn = attn[feat_N:, :feat_N]
            heads_heads_attn = attn[feat_N:, feat_N:]
            
            attn_dicts['feats_feats_attn'] = feats_feats_attn
            attn_dicts['feats_heads_attn'] = feats_heads_attn
            attn_dicts['heads_feats_attn'] = heads_feats_attn
            attn_dicts['heads_heads_attn'] = heads_heads_attn
            attn_dicts['self_heads_attn'] = torch.diag(heads_heads_attn, 0)
            
            attn_list.append(attn_dicts)
        
        return attn_list    #* [attn_dict1, attn_dict2] / attn_dict={inter_feats, feats_heads}