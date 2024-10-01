
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dual_Classifier(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(Dual_Classifier, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        #todo create dual classifier
        
        self.dual_classifier = nn.Linear(self.embedding_size, self.num_classes*2, bias=False)
        nn.init.normal_(self.dual_classifier.weight)
        
        self.eps = 1e-8
    
    
    #todo 1. Known - Known Independent (Loss: KKI)
    #todo 2. Known - Unknowns  Independent (Loss: KUI)
    #todo 3. Unkonwn (Mixed data) Handler is required..
    
    def forward(self, feat):
        out = self.dual_classifier(feat)    #* B, num_CLS * 2
        # a_dual = torch.stack([i for i in a.chunk(2, dim=1)]).permute(1,0,2)
        # out_dual = torch.stack([o for o in out.chunk(2, dim=1)]).permute(1,0,2) #* B, 2 (Known, Unknown), num_CLS
        out_dual = out.view(out.size(0), 2, -1)
        # if prob:
        #     prob = out_dual.softmax(dim=-1)
        return out_dual
    
    def known_ce_loss(self, out, label):
        # out_open = out_open.view(out_s.size(0), 2, -1)
        # open_loss_pos, open_loss_neg = ova_loss(out_open, label_s)
        assert len(out.size()) == 3
        assert out.size(1) == 2

        sout = F.softmax(out, 1)
        label_p = torch.zeros((out.size(0), out.size(2))).long().cuda()
        label_range = torch.arange(0, out.size(0)).long()
        label_p[label_range, label.long()] = 1
        label_n = 1 - label_p
        
        ce_loss = F.cross_entropy(out[:,0], label.long())
        
        dual_ce_loss_pos = torch.mean(torch.sum(-torch.log(sout[:, 0, :] + self.eps) * label_p, 1))
        dual_ce_loss_neg = torch.mean(torch.max(-torch.log(sout[:, 1, :] + self.eps) * label_n, 1)[0])
        
        # dual_ce_loss_pos = F.nll_loss(sout[:,0],label_p)
        # dual_ce_loss_neg = F.nll_loss(sout[:,1],label_n)
        
        return ce_loss, dual_ce_loss_pos, dual_ce_loss_neg
    
    def mixed_ce_loss(self, mixed_out, mixed_label):
        #todo 1. mixed_Label = [0., 0., 0.7, ... 0., 0.3, 0.]
        #! Unknown Prob --> CE with Mixed Label
        #! Increase Unknown prob to discern similar imaginary samples
        #! --> Increase Discriminative power!
        
        # mixed_out = F.softmax(mixed_out, 1)
        # mix_ce_loss = torch.mean(torch.sum(-torch.log(mixed_out[:, 1, :] + self.eps) * mixed_label, 1))
        mix_ce_loss = F.cross_entropy(mixed_out[:, 1], mixed_label)
        
        return mix_ce_loss
    
    def known_unknown_contrastive_loss(self):
        #todo 1. Known - Known Independent (Loss: KKI)
        #todo 2. Known - Unknowns  Independent (Loss: KUI)
        #* Knowns: self.dual_classifier[:self.num_classes]
        #* Unknowns: self.dual_classifier[self.num_classes:]
        
        k_vecs = self.dual_classifier.weight[:self.num_classes]
        unk_vecs = self.dual_classifier.weight[self.num_classes:]
        norm_k = F.normalize(k_vecs, dim=-1)
        norm_unk = F.normalize(unk_vecs, dim=-1)
        
        #? Knowns-Knowns --> Independent
        kk_aff_mat = F.linear(norm_k, norm_k)  #* Affinity Matrix (Cosine)
        # label_range = torch.arange(0, 160).long()
        kk_aff_labels = torch.arange(norm_k.shape[0], device='cuda').long()
        
        loss_KKI = F.cross_entropy(kk_aff_mat, kk_aff_labels)
        
        #? Knowns-Unknowns --> independent
        ku_aff_mat = F.linear(norm_k, norm_unk)
        loss_KUI = ku_aff_mat.sum(dim=1).mean()
        
        return loss_KKI, loss_KUI
    
    def get_split_indices(self, feats, thres=0.9):
        self.eval()
        outs = self.forward(feats)   #* B, 2 (Known, Unknown), num_CLS
        # outs_vals, outs_idxs = F.softmax(outs, 1)[:, 0].max(dim=1)   #* B, num_CLS
        outs_vals, outs_idxs = F.softmax(outs[:, 0], 1).max(dim=1)   #* B, num_CLS
        
        idx = torch.where(outs_vals>thres, 1, 0)
        
        k_idx = torch.nonzero(idx).squeeze().cpu()
        unk_idx = torch.nonzero(1-idx).squeeze().cpu()
        # for sample_idx, (out_v, out_i) in enumerate(zip(outs_vals, outs_idxs)):
        #     torch.where()
        k_label = outs_idxs[k_idx]
        self.train()
        return k_idx, k_label, unk_idx
    
    def get_predicts(self, feats):
        self.eval()
        out = self.forward(feats)
        # sout = F.softmax(out, 1)[:,0] #* get known section B, num_CLS
        sout = F.softmax(out[:,0], 1) #* get known section B, num_CLS
        _, preds = sout.max(dim=1)
        self.train()
        return preds
    
    
    def entropy(self):
        pass
    
    
    def rebuild_classifier(self, nb_classes_now):
        # self.dual_classifier = nn.Linear(self.embedding_size, self.num_classes*2, bias=False)
        # nn.init.normal_(self.dual_classifier.weight)
        old_classes, _ = self.dual_classifier.weight.data.shape
        old_wts = self.dual_classifier.weight.data.clone()
        del self.dual_classifier
        self.dual_classifier = nn.Linear(self.embedding_size, nb_classes_now*2, bias=False)
        nn.init.normal_(self.dual_classifier.weight)
        self.dual_classifier.weight.data[:old_classes] = old_wts
        
    def show_shape(self):
        print('Dual Head Wts:',self.dual_classifier.weight.shape)
        