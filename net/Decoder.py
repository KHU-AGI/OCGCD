
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Ours_utils.utils import *
class Head(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(Head, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        #todo create dual classifier
        
        self.head = nn.Linear(self.embedding_size, self.num_classes)
        # nn.init.normal_(self.head.weight)
        # nn.init.normal_(self.head.bias)
        
        self.eps = 1e-8
    
    def forward(self, feat):
        out = self.head(feat)
        return out
    
    def update(self, nb_classes_now):
        old_classes, _ = self.head.weight.data.shape
        old_wts = self.head.weight.data.clone()
        old_bias = self.head.bias.data.clone()
        del self.head
        self.head = nn.Linear(self.embedding_size, nb_classes_now)
        # nn.init.normal_(self.head.weight)
        # nn.init.normal_(self.head.bias)
        self.head.weight.data[:old_classes] = old_wts
        self.head.bias.data[:old_classes] = old_bias
        
        self.seen_classes = old_classes
    
    
    # def show_shape(self):
    #     print('Dual Head Wts:',self.dual_classifier.weight.shape)
        

from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture

class Pseudo_Module(nn.Module):
    def __init__(self, sz_feature=768, low_scale=4):
        super(Pseudo_Module, self).__init__()
        self.fc_down = nn.Linear(sz_feature, sz_feature // low_scale)
        # self.batch2 = torch.nn.BatchNorm1d(sz_feature // low_scale)
        self.act = nn.GELU()
        self.fc_up = nn.Linear(sz_feature // low_scale, sz_feature)

    def forward(self, X):
        out_f = self.fc_down(X)
        out_f = self.act(out_f)
        out_f = self.fc_up(out_f)
        return out_f

    
    def clustering_unknown(self, ex_model_, unknown_dset, known_dset, sspl_epochs, args):
        self.SS_Pseudo_training(ex_model_, unknown_dset, known_dset, sspl_epochs, nb_workers=args.nb_workers, lr= args.lr)  #* args.lr*10
        un_dlod = torch.utils.data.DataLoader(unknown_dset, batch_size=len(unknown_dset), shuffle=False, num_workers=args.nb_workers)
        ex_model_.eval()
        self.eval()
        with torch.no_grad():
            un_feats, _, _ = extract_embedding(ex_model_, un_dlod)
            m_un_feats = self.forward(un_feats)
        
        clst_a = AffinityPropagation(random_state=args.seed).fit(m_un_feats.cpu().numpy()) # 0.75
        p, c = np.unique(clst_a.labels_, return_counts=True)
        nb_classes_k = len(p)
        print("nb_classes_k:", nb_classes_k)
        # pseudo_unseen = clst_a.labels_
        ####
        # if args.use_GM_clustering:
        gm = GaussianMixture(n_components=nb_classes_k, max_iter=1000, tol=1e-4, init_params='kmeans', random_state=args.seed).fit(m_un_feats.cpu().numpy()) 
        pseudo_unseen = gm.predict(m_un_feats.cpu().numpy())
        
        return nb_classes_k, pseudo_unseen

    def SS_Pseudo_training(self, ex_model_, unknown_dset, known_dset, sspl_epochs, nb_workers=2, lr=1e-4, weight_decay=5e-3):
        #todo 1. module Training 
        #todo 2. Unlabeled Pseudo Labeling via trained module
        ex_model_ = copy.deepcopy(ex_model_)
        ex_model_ = ex_model_.cuda()
        ex_model_.eval()
        self.train()
        param_groups = [{'params': self.parameters()}]
        opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        unknown_dset.extra_trsf = weak_trsf_pairwise()
        un_dlod = torch.utils.data.DataLoader(unknown_dset, batch_size=len(unknown_dset), shuffle=False, num_workers=nb_workers)
        kn_dlod = torch.utils.data.DataLoader(known_dset, batch_size=len(known_dset), shuffle=False, num_workers=nb_workers)
        for ss_epoch in range(sspl_epochs):
            with torch.no_grad():
                un_feats, _, _ = extract_embedding(ex_model_, un_dlod, multi_view=True)
                kn_feats, _, _ = extract_embedding(ex_model_, kn_dlod)
                
            m_un_feats = self.forward(un_feats)
            m_kn_feats = self.forward(kn_feats)
            
            #* logits, labels = self.ssl_func(m_un_feats, m_kn_feats)
            logits, labels = self.ssl_func(m_un_feats)
            # print('[Pseudo Module SSPL] logits:',logits.shape)
            # print('[Pseudo Module SSPL] labels:',labels.shape)
            ss_loss = F.cross_entropy(logits, labels)
            
            opt.zero_grad()
            ss_loss.backward()
            opt.step()
            
            print('[Pseudo Module SSPL][Epoch/Epochs: {}/ {}] SS_Loss: {}'.format(ss_epoch, sspl_epochs, ss_loss.item()))
        self.eval()
        unknown_dset.extra_trsf = False

    # def ssl_func(self, features, known_feats):
    #     #todo Self-Supervised Learning
    #     temperature = 1.0
    #     b_ = 0.5 * int(features.size(0))
    #     feat_sz = features.shape[0]

    #     labels = torch.cat([torch.arange(b_) for i in range(2)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.cuda()
    #     labels = torch.cat([labels, torch.zeros(feat_sz, known_feats.shape[0], device='cuda')], dim=1)

    #     similarity_matrix_ori = torch.matmul(features, torch.cat([features, known_feats]).T)

    #     #* discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    #     labels = labels[:,:feat_sz][~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix_ori[:,:feat_sz][~mask].view(similarity_matrix_ori.shape[0], -1)

    #     #* select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    #     # print("positives:",positives.shape); print()

    #     #* select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    #     negatives = torch.cat([negatives, similarity_matrix_ori[:, feat_sz:]], dim=-1)
    #     # print("negatives:", negatives.shape); print()

    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    #     logits = logits / temperature
    #     return logits, labels
    
    def ssl_func(self, features):
        b_ = 0.5 * int(features.size(0))

        labels = torch.cat([torch.arange(b_) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device='cuda')
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device='cuda')

        # logits = logits / args.temperature
        return logits, labels

    # def loss_func(self, known_feats, unknown_feats):
    #     D_feats = torch.cat([known_feats, unknown_feats], dim=0)
    #     known_labels = torch.zeros(known_feats.shape[0], device='cuda')
    #     unknown_labels = torch.ones(unknown_feats.shape[0], device='cuda')
    #     D_labels = torch.cat([known_labels, unknown_labels], dim=0)
        
    #     labels = torch.zeros((D_labels.shape[0], 2)).long().cuda()
    #     label_range = torch.arange(0, D_labels.shape[0]).long()
    #     labels[label_range, D_labels.long()] = 1
    #     # print()
    #     # print(labels)
    #     # print()
        
    #     r_idx = torch.randperm(labels.shape[0])
        
    #     D_logits = self.forward(D_feats[r_idx])
    #     # loss = F.binary_cross_entropy_with_logits(D_logits, D_labels[r_idx])
    #     loss = -torch.mean(torch.sum(F.log_softmax(D_logits, dim=1) * labels[r_idx], dim=1))
        
    #     return loss

    # def discriminate_func(self, feats, thres=0.9):
    #     self.eval()
    #     outs = self.forward(feats)   #* B, 2 (Known, Unknown), num_CLS
        
    #     _, preds = outs.max(dim=1)
        
    #     idx = torch.where(preds==0, 0, 1)
    #     unk_idx = torch.nonzero(idx).squeeze().cpu()
    #     k_idx = torch.nonzero(1-idx).squeeze().cpu()
    #     return unk_idx, k_idx
        