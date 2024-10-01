import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import sklearn.preprocessing

def Sup_Con_Loss(features, labels=None, mask=None, temperature=0.07, base_temperature=0.07):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """

    device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),temperature)

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss




class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return [out1, out2]

def weak_trsf_pairwise():
    resnet_sz_resize = 256
    resnet_sz_crop = 224 
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    transform = TransformTwice(transforms.Compose([
            transforms.RandomResizedCrop(resnet_sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(resnet_sz_resize),
            transforms.CenterCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std)
            ]))
    return transform

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes = range(0, nb_classes))
    T = torch.FloatTensor(T).cuda()
    return T


def info_nce_logits(features):
    #todo Self-Supervised Learning
    temperature = 1.0
    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    #* assert similarity_matrix.shape == (
    #*     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    #* assert similarity_matrix.shape == labels.shape

    #* discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #* assert similarity_matrix.shape == labels.shape

    #* select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    #* select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / temperature
    return logits, labels


def head_feat_contrastive_loss(features, labels, heads):
    # labels = torch.tensor(labels.clone().detach()).repeat(2,)
    # labels = labels.clone().detach().repeat(2,)
    margin = 0.1
    scale = 32.
    cos = F.linear(features, F.normalize(heads, dim=-1))  # Calcluate cosine similarity

    P_one_hot = binarize(T = labels, nb_classes = heads.shape[0])
    N_one_hot = 1 - P_one_hot

    pos_exp = torch.exp(-scale*(cos - margin))
    neg_exp = torch.exp(scale*(cos + margin))

    with_pos_heads = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
    num_valid_heads = len(with_pos_heads)   # The number of positive proxies
    
    P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
    N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

    if num_valid_heads == 0:
        num_valid_heads = 1
    pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_heads
    neg_term = torch.log(1 + N_sim_sum).sum() / heads.shape[0]
    loss = pos_term + neg_term

    return loss

def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


def get_ETF_Head(num_classes, in_channels):
    with torch.no_grad():
        orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)
        i_nc_nc = torch.eye(num_classes)
        one_nc_nc = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(num_classes / (num_classes - 1)))
        etf_vec = etf_vec.t()
        etf_vec.requires_grad=False
    # print(etf_vec.shape)    #* num_classes, in_channels
    return etf_vec

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p, dim=1)
    en = -torch.sum(p * torch.log(p+1e-8), 1)
    if mean:
        return torch.mean(en)
    else:
        return en

def ETF_discriminate_loss(etf_feats, etf_head, ori_label, mixed_label, lam, nb_classes):
    margin=0.1
    scale=32.
    #? loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    # ori_x, _ = [sp_x for sp_x in x.chunk(2)]
    ori_feats, mixed_feats = [sp_feat for sp_feat in etf_feats.chunk(2)]
    # known_area, unknown_area = etf_head[:nb_classes], etf_head[nb_classes:]
    #todo Case 1) Original feat - Known Area : Positive
    ori_cos = F.linear(ori_feats, etf_head)
    ori_exp = torch.exp(scale*(ori_cos - margin)) #* Batch x classes
    # print('[ETF-Loss] ori_exp:', ori_exp.shape)
    ori_known_logit, ori_unknown_logit = ori_exp[:,:nb_classes], ori_exp[:,nb_classes:]
    # print('[ETF-Loss] ori_unknown_logit:', ori_unknown_logit.shape)
    # print('[ETF-Loss] ori_unknown_logit:', ori_unknown_logit.shape)
    
    ce_known = F.cross_entropy(ori_known_logit, ori_label)
    ent_ori_unknown = entropy(ori_unknown_logit)
    
    # known_loss = ce_known - ent_unknown
    # known_loss = ce_known/ent_unknown
    
    #todo Case 2) Mixed feat - Unknown Area : Positive
    mixed_cos = F.linear(mixed_feats, etf_head)
    mixed_logit = torch.exp(scale*(mixed_cos - margin)) #* Batch x classes
    # print('[ETF-Loss] mixed_logit:', mixed_logit.shape)
    # mixed_known_logit, mixed_unknown_logit = mixed_exp[:,:nb_classes], mixed_exp[:,nb_classes:]
    mixed_known_logit = mixed_logit[:,:nb_classes]
    # print('[ETF-Loss] mixed_known_logit:', mixed_known_logit.shape)
    #* ce_unknown = F.cross_entropy(mixed_logit, ori_label+nb_classes) * lam + F.cross_entropy(mixed_logit, mixed_label+nb_classes) * (1. - lam)
    ent_mixed_known = entropy(mixed_known_logit)
    
    #* unknown_loss = ce_unknown - ent_known
    # unknown_loss = ce_unknown/ent_known
    
    #todo Case 3) Original feat - All the Mixed feat : Negative
    # neg_feats_cos = F.linear(ori_feats, mixed_feats)
    # neg_feats_exp = torch.exp(scale*(neg_feats_cos + margin)) #* Batch x Batch
    # return ce_known, ent_ori_unknown, ent_mixed_known
    return ce_known, ent_mixed_known

    #! pos_exp = torch.exp(-scale*(cos - margin))
    #! neg_exp = torch.exp(scale*(cos + margin))
    
    #!P_one_hot = binarize(T = labels, nb_classes = heads.shape[0])
    #!N_one_hot = 1 - P_one_hot
    #!P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
    #!N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
    
    #! pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_heads
    #! neg_term = torch.log(1 + N_sim_sum).sum() / heads.shape[0]
    # pass


#* Cutmix from the CutMix-PyTorch
#* (https://github.com/clovaai/CutMix-PyTorch/tree/2d8eb68faff7fe4962776ad51d175c3b01a25734)
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_cut_mix(samples, targets, beta=1.0):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(samples.size()[0]).cuda()
    target_a = targets
    target_b = targets[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(samples.size(), lam)
    samples[:, :, bbx1:bbx2, bby1:bby2] = samples[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (samples.size()[-1] * samples.size()[-2]))
    #? loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    return samples, target_a, target_b, lam

from torch.autograd import Variable
def get_mix_up(x, y, alpha=1.0):
    #* Cutmix from the FAIR
    #* (https://github.com/facebookresearch/mixup-cifar10/tree/eaff31ab397a90fbc0a4aac71fb5311144b3608b)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    
    mixed_x, y_a, y_b = map(Variable, (mixed_x, y_a, y_b))
    
    return mixed_x, y_a, y_b, lam

import random
# random.seed(seed)
def cutmix_or_mixup(x,y):
    transforms=['mixup', 'cut_mix']
    chosen = random.choice(transforms)
    if chosen == 'mixup':
        mixed_x, y_a, y_b, lam = get_mix_up(x, y, alpha=1.0)
    else: #*'cut_mix'
        mixed_x, y_a, y_b, lam = get_cut_mix(x, y, beta=1.0)
    
    return mixed_x, y_a, y_b, lam


def extract_embedding(model, dataloader, etf_embed=False):
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in dataloader:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    if etf_embed:
                        _, J = model(J.cuda(), etf_embedding=etf_embed)
                    else:
                        J = model(J.cuda())
                    # J, _ = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    # model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]


# utils.show_OnN(feats, labels, preds_cs, nb_classes_prv, inc_pth_rst_exp, thres, True)
# def discover_acc(m, y, v, nb_classes, pth_result, thres=0., is_hist=False, iter=0):
def discover_acc(known_sim, unknown_sim, labels, nb_classes):
    oo_i, on_i, no_i, nn_i = 0, 0, 0, 0
    # o, n = [], []
    
    # known_sim,_ = cos_sim[:,:nb_classes].max(dim=1)
    # unknown_sim,_ = cos_sim[:,nb_classes:].max(dim=1)

    for j in range(len(labels)):
        if labels[j] < nb_classes:
            # o.append(cos_sim[j].cpu().numpy())
            if known_sim[j] >= unknown_sim[j]:
                oo_i += 1
            else:
                on_i += 1
        else:
            # n.append(cos_sim[j].cpu().numpy())
            if known_sim[j] >= unknown_sim[j]:
                no_i += 1
            else:
                nn_i += 1

    # if is_hist is True:
    #     plt.hist((o, n), histtype='bar', bins=100)
    #     plt.savefig(pth_result + '/' + 'Init_Split_' + str(iter) + '.png')
    #     plt.close()
        # plt.clf()

    print('ETF-Based Discovery\t oo: {}\t on: {}\t no: {}\t nn: {}'.format(oo_i, on_i, no_i, nn_i))
    discover_score={}
    discover_score['old_detect_acc'] = oo_i / (oo_i+on_i)
    discover_score['new_detect_acc'] = nn_i / (no_i+nn_i)
    discover_score['all_detect_acc'] = (oo_i+nn_i) / (oo_i+on_i+no_i+nn_i)
    return discover_score
    
















def get_grad_norm(model, report_dict, args):
    grad={}
    if args.prompt_tuning_layers is not None:
        grad['prompt'] = torch.norm(model.backbone.prompts.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
        # qkv_li = []
        # for layer_idx in args.prompt_tuning_layers:
        #     qkv_li.append(torch.norm(model.backbone.blocks[layer_idx].attn.qkv.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().unsqueeze(0))
            
        # qkv_li = torch.cat(qkv_li, dim=0)
        # grad['attn_qkv'] = qkv_li.mean().cpu()
        
    elif args.lora_layers is not None:
        lora_li = []
        # qkv_li = []
        for layer_idx in args.lora_layers:
            # lora_a1, lora_a2 = [torch.norm(model.backbone.blocks[layer_idx].attn.kv_lora.lora_A[i].grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu() for i in range(2)]
            # lora_b1, lora_b2 = [torch.norm(model.backbone.blocks[layer_idx].attn.kv_lora.lora_B[i].grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu() for i in range(2)]
            lora_a1, lora_a2 = [torch.norm(model.backbone.blocks[layer_idx].attn.kv_lora.lora_A[i].weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu() for i in range(2)]
            lora_b1, lora_b2 = [torch.norm(model.backbone.blocks[layer_idx].attn.kv_lora.lora_B[i].weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu() for i in range(2)]
            
            lora_li.append(torch.tensor([lora_a1, lora_a2, lora_b1, lora_b2]).mean().unsqueeze(0))
            
            # qkv_li.append(torch.norm(model.backbone.blocks[layer_idx].attn.qkv.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().unsqueeze(0))
            
        # qkv_li = torch.cat(qkv_li, dim=0)
        lora_li = torch.cat(lora_li, dim=0)
        
        grad['lora'] = lora_li.mean().cpu()
        # grad['attn_qkv'] = qkv_li.mean().cpu()
        
    elif args.adapt_mlp_layers is not None:
        adapt_mlp_li=[]
        # mlp_li=[]
        for layer_idx in args.adapt_mlp_layers:
            adapter_down = torch.norm(model.backbone.blocks[layer_idx].a_mlp.down_proj.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
            adapter_up = torch.norm(model.backbone.blocks[layer_idx].a_mlp.up_proj.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
            adapt_mlp_li.append(torch.tensor([adapter_down, adapter_up]).mean().unsqueeze(0))
            
            # mlp_fc1 = torch.norm(model.backbone.blocks[layer_idx].mlp.fc1.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
            # mlp_fc2 = torch.norm(model.backbone.blocks[layer_idx].mlp.fc2.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
            # mlp_li.append(torch.tensor([mlp_fc1, mlp_fc2]).mean().unsqueeze(0))
        
        adapt_mlp_li = torch.cat(adapt_mlp_li, dim=0)
        # mlp_li = torch.cat(mlp_li, dim=0)
        
        grad['adapt_mlp_li'] = adapt_mlp_li.mean().cpu()
        # grad['mlp'] = mlp_li.mean().cpu()
        
    elif args.adapter_layers is not None:
        adapter_li=[]
        # mlp_li=[]
        for layer_idx in args.adapter_layers:
            adapter_down = torch.norm(model.backbone.blocks[layer_idx].adapter.down_proj.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
            adapter_up = torch.norm(model.backbone.blocks[layer_idx].adapter.up_proj.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
            adapter_li.append(torch.tensor([adapter_down, adapter_up]).mean().unsqueeze(0))
            
            # mlp_fc1 = torch.norm(model.backbone.blocks[layer_idx].mlp.fc1.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
            # mlp_fc2 = torch.norm(model.backbone.blocks[layer_idx].mlp.fc2.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
            # mlp_li.append(torch.tensor([mlp_fc1, mlp_fc2]).mean().unsqueeze(0))
        
        adapter_li = torch.cat(adapter_li, dim=0)
        # mlp_li = torch.cat(mlp_li, dim=0)
        
        grad['adapter'] = adapter_li.mean().cpu()
        # grad['mlp'] = mlp_li.mean().cpu()
    else:
        # qkv_li = []
        # mlp_li=[]
        # for layer_idx in range(len(model.backbone.blocks)):
        #     qkv_li.append(torch.norm(model.backbone.blocks[layer_idx].attn.qkv.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().unsqueeze(0))
        #     mlp_fc1 = torch.norm(model.backbone.blocks[layer_idx].mlp.fc1.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
        #     mlp_fc2 = torch.norm(model.backbone.blocks[layer_idx].mlp.fc2.weight.grad.clone().detach(), p=2, dim=-1).reshape(-1,).mean().cpu()
        #     mlp_li.append(torch.tensor([mlp_fc1, mlp_fc2]).mean().unsqueeze(0))
        
        # qkv_li = torch.cat(qkv_li, dim=0)
        # mlp_li = torch.cat(mlp_li, dim=0)
        
        # grad['attn_qkv'] = qkv_li.mean().cpu()
        # grad['mlp'] = mlp_li.mean().cpu()
        pass
    
    
    # return grad
    report_dict['grad_norm'].append(grad)


def get_fisher_information(model, report_dict, args):
    if args.prompt_tuning_layers is not None:
        attn_fisher={}
        for layer_idx in args.prompt_tuning_layers:
            for n, p in model.backbone.blocks[layer_idx].named_parameters():
                if "attn.qkv.weight" in n:
                    attn_fisher[n]=torch.zeros(p.shape)
        prefix_fisher = {
            'prefix': torch.zeros(model.backbone.prompts.shape)
        }

        for layer_idx in args.prompt_tuning_layers:
            for n,p in model.backbone.blocks[layer_idx].named_parameters():
                if "attn.qkv.weight" in n:
                    attn_fisher[n] += p.grad.pow(2).clone().detach().cpu()
        
        for n, p in attn_fisher.items():
            # attn_fisher[n] = p / len(n_samples)
            attn_fisher[n] = torch.min(attn_fisher[n], torch.tensor(1e-3)).mean()
        
        prefix_fisher['prefix'] += model.backbone.prompts.grad.pow(2).clone().detach().cpu()
        prefix_fisher['prefix'] = torch.min(prefix_fisher['prefix'], torch.tensor(1e-3)).mean()
        
        report_dict['fisher_ViT'] = attn_fisher
        report_dict['fisher_PET'] = prefix_fisher
        
    elif args.lora_layers is not None:
        attn_fisher={}
        for layer_idx in args.lora_layers:
            for n, p in model.backbone.blocks[layer_idx].named_parameters():
                if "attn.qkv.weight" in n:
                    attn_fisher[n]=torch.zeros(p.shape)
        
        lora_fisher={}
        for layer_idx in args.lora_layers:
            for n, p in model.backbone.blocks[layer_idx].named_parameters():
                if "attn.kv_lora.lora_A." in n or "attn.kv_lora.lora_B." in n:
                    lora_fisher[n]=torch.zeros(p.shape)
        
        for layer_idx in args.lora_layers:
            for n,p in model.backbone.blocks[layer_idx].named_parameters():
                if "attn.qkv.weight" in n:
                    attn_fisher[n] += p.grad.pow(2).clone().detach().cpu()
                
                if "attn.kv_lora.lora_A." in n or "attn.kv_lora.lora_B." in n:
                    lora_fisher[n] += p.grad.pow(2).clone().detach().cpu()
        
        for n, p in attn_fisher.items():
            attn_fisher[n] = torch.min(attn_fisher[n], torch.tensor(1e-3)).mean()
        
        for n, p in lora_fisher.items():
            lora_fisher[n] = torch.min(lora_fisher[n], torch.tensor(1e-3)).mean()
        
        report_dict['fisher_ViT'] = attn_fisher
        report_dict['fisher_PET'] = lora_fisher
        
    elif args.adapt_mlp_layers is not None:
        mlp_fisher={}
        for layer_idx in args.adapt_mlp_layers:
            for n, p in model.backbone.blocks[layer_idx].named_parameters():
                if ".fc1.weight" in n or ".fc2.weight" in n:
                    mlp_fisher[n]=torch.zeros(p.shape)
        
        adaptmlp_fisher={}
        for layer_idx in args.adapt_mlp_layers:
            for n, p in model.backbone.blocks[layer_idx].named_parameters():
                if "a_mlp.down_proj" in n or "a_mlp.up_proj" in n:
                    adaptmlp_fisher[n]=torch.zeros(p.shape)
        
        for layer_idx in args.adapt_mlp_layers:
            for n,p in model.backbone.blocks[layer_idx].named_parameters():
                if ".fc1.weight" in n or ".fc2.weight" in n:
                    mlp_fisher[n] += p.grad.pow(2).clone().detach().cpu()
                
                if "a_mlp.down_proj" in n or "a_mlp.up_proj" in n:
                    adaptmlp_fisher[n] += p.grad.pow(2).clone().detach().cpu()
        
        for n, p in mlp_fisher.items():
            mlp_fisher[n] = torch.min(mlp_fisher[n], torch.tensor(1e-3)).mean()
        
        for n, p in adaptmlp_fisher.items():
            adaptmlp_fisher[n] = torch.min(adaptmlp_fisher[n], torch.tensor(1e-3)).mean()
        
        report_dict['fisher_ViT'] = mlp_fisher
        report_dict['fisher_PET'] = adaptmlp_fisher
        
    elif args.adapter_layers is not None:
        mlp_fisher={}
        for layer_idx in args.adapter_layers:
            for n, p in model.backbone.blocks[layer_idx].named_parameters():
                if ".fc1.weight" in n or ".fc2.weight" in n:
                    mlp_fisher[n]=torch.zeros(p.shape)
        
        adapter_fisher={}
        for layer_idx in args.adapter_layers:
            for n, p in model.backbone.blocks[layer_idx].named_parameters():
                if "adapter.down_proj" in n or "adapter.up_proj" in n:
                    adapter_fisher[n]=torch.zeros(p.shape)
        
        for layer_idx in args.adapter_layers:
            for n,p in model.backbone.blocks[layer_idx].named_parameters():
                if ".fc1.weight" in n or ".fc2.weight" in n:
                    mlp_fisher[n] += p.grad.pow(2).clone().detach().cpu()
                
                if "adapter.down_proj" in n or "adapter.up_proj" in n:
                    adapter_fisher[n] += p.grad.pow(2).clone().detach().cpu()
        
        for n, p in mlp_fisher.items():
            mlp_fisher[n] = torch.min(mlp_fisher[n], torch.tensor(1e-3)).mean()
        
        for n, p in adapter_fisher.items():
            adapter_fisher[n] = torch.min(adapter_fisher[n], torch.tensor(1e-3)).mean()
        
        report_dict['fisher_ViT'] = mlp_fisher
        report_dict['fisher_PET'] = adapter_fisher
    else:
        mlp_fisher={}
        for layer_idx in range(len(model.backbone.blocks)):
            for n, p in model.backbone.blocks[layer_idx].named_parameters():
                if ".fc1.weight" in n or ".fc2.weight" in n:
                    mlp_fisher[n]=torch.zeros(p.shape)
        
        attn_fisher={}
        for layer_idx in range(len(model.backbone.blocks)):
            for n, p in model.backbone.blocks[layer_idx].named_parameters():
                if "attn.qkv.weight" in n:
                    attn_fisher[n]=torch.zeros(p.shape)
        
        
        for layer_idx in range(len(model.backbone.blocks)):
            for n,p in model.backbone.blocks[layer_idx].named_parameters():
                if ".fc1.weight" in n or ".fc2.weight" in n:
                    mlp_fisher[n] += p.grad.pow(2).clone().detach().cpu()
                if "attn.qkv.weight" in n:
                    attn_fisher[n] += p.grad.pow(2).clone().detach().cpu()
        
        for n, p in mlp_fisher.items():
            mlp_fisher[n] = torch.min(mlp_fisher[n], torch.tensor(1e-3)).mean()
        for n, p in attn_fisher.items():
            attn_fisher[n] = torch.min(attn_fisher[n], torch.tensor(1e-3)).mean()
        
        report_dict['fisher_ViT_MLP'] = mlp_fisher
        report_dict['fisher_ViT_ATTN'] = attn_fisher
        # raise Exception("can not find PET Modules..")
    

def get_retain_grads(model,args):
    if args.prompt_tuning_layers is not None:
        for layer_idx in args.prompt_tuning_layers:
            model.backbone.blocks[layer_idx].attn.qkv.weight.retain_grad()
        
    elif args.lora_layers is not None:
        for layer_idx in args.lora_layers:
            model.backbone.blocks[layer_idx].attn.qkv.weight.retain_grad()
            
    elif args.adapt_mlp_layers is not None:
        adapt_mlp_li=[]
        mlp_li=[]
        for layer_idx in args.adapt_mlp_layers:
            model.backbone.blocks[layer_idx].mlp.fc1.weight.retain_grad()
            model.backbone.blocks[layer_idx].mlp.fc2.weight.retain_grad()

    elif args.adapter_layers is not None:
        for layer_idx in args.adapter_layers:
            model.backbone.blocks[layer_idx].mlp.fc1.weight.retain_grad()
            model.backbone.blocks[layer_idx].mlp.fc2.weight.retain_grad()
    else:
        for layer_idx in range(len(model.backbone.blocks)):
            model.backbone.blocks[layer_idx].attn.qkv.weight.retain_grad()
            model.backbone.blocks[layer_idx].mlp.fc1.weight.retain_grad()
            model.backbone.blocks[layer_idx].mlp.fc2.weight.retain_grad()
        # raise Exception("can not find PET Modules..")

#! Fourier
#!====================================================
import math
import glob
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from einops import rearrange, reduce, repeat


def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))


def make_segments(x, y):  # make segment for `plot_segment`
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_segment(ax, xs, ys, cmap_name="plasma"):  # plot with cmap segments
    z = np.linspace(0.0, 1.0, len(ys))
    z = np.asarray(z)
    
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(0.0, 1.0)
    segments = make_segments(xs, ys)
    lc = LineCollection(segments, array=z, cmap=cmap_name, norm=norm,
                        linewidth=2.5, alpha=1.0)
    ax.add_collection(lc)

    colors = [cmap(x) for x in xs]
    ax.scatter(xs, ys, color=colors, marker="o", zorder=100)


def get_fourier_latents(report_path, latent_path):
    fourier_path = os.path.join(report_path,'Fourier')
    os.makedirs(fourier_path, exist_ok=True)
    
    dict_paths = glob.glob(f'{latent_path}/*.pth')
    for idx, dict_path in enumerate(dict_paths):
        # path = os.path.join(dict_path,'report_dict.pth')
        report_dict = torch.load(dict_path)
        #*latents = report_dict['img_latents'].cpu()
        latents = report_dict.cpu()
    # Fourier transform feature maps
        fourier_latents = []
        for l_idx, latent in enumerate(latents):  # `latents` is a list of hidden feature maps in latent spaces
            # latent = latent.cpu()
            
            if len(latent.shape) == 3:  # for ViT
                b, n, c = latent.shape
                h, w = int(math.sqrt(n)), int(math.sqrt(n))
                latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
            elif len(latent.shape) == 4:  # for CNN
                b, c, h, w = latent.shape
            else:
                raise Exception("shape: %s" % str(latent.shape))
            latent = fourier(latent)
            # if l_idx ==0:
            #     print('fourier shape:', latent.shape)
            latent = shift(latent)
            # if l_idx ==0:
            #     print('shift shape:', latent.shape)
            latent = shift(latent).mean(dim=(0, 1))
            # if l_idx ==0:
            #     print('shift(latent).mean(dim=(0, 1)) shape:', latent.shape)
            latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
            latent = latent - latent[0]  # visualize 'relative' log amplitudes 
                                        # (i.e., low-freq amp - high freq amp)
            # if l_idx ==0:
            #     print('latent = latent - latent[0] shape:', latent.shape)
            fourier_latents.append(latent)
            
        fourier_latents = torch.stack(fourier_latents)
        if idx ==0:
            print('fourier_latents = torch.stack(fourier_latents) shape:', fourier_latents.shape)
        torch.save(fourier_latents, f'{fourier_path}/Fourier_latent_bs{idx}.pth')
#!====================================================

def get_latents_variance(report_path, latent_path):
    variance_path = os.path.join(report_path,'Variance')
    os.makedirs(variance_path, exist_ok=True)
    
    dict_paths = glob.glob(f'{latent_path}/*.pth')
    for idx, dict_path in enumerate(dict_paths):
        # path = os.path.join(dict_path,'report_dict.pth')
        report_dict = torch.load(dict_path)
        #*latents = report_dict['img_latents'].cpu()
        latents = report_dict.cpu()
        variances = []
        for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
            latent = latent.cpu()
            
            if len(latent.shape) == 3:  # for ViT
                b, n, c = latent.shape
                h, w = int(math.sqrt(n)), int(math.sqrt(n))
                latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
            elif len(latent.shape) == 4:  # for CNN
                b, c, h, w = latent.shape
            else:
                raise Exception("shape: %s" % str(latent.shape))
                        
            variances.append(latent.var(dim=[-1, -2]).mean(dim=[0, 1]))
            
        variances = torch.stack(variances)
        if idx ==0:
            print('variances = torch.stack(variances) shape:', variances.shape)
        torch.save(variances, f'{variance_path}/Variance_latent_bs{idx}.pth')
#!====================================================

#* Original
#* ===============================================================
def Attn_Contrastive_Loss(a_feats, labels, a_heads):
    mask = torch.zeros(len(a_heads), dtype=torch.bool)
    mask[torch.unique(labels)] = True

    pos_head_idx = torch.nonzero(mask, as_tuple=True)[0]
    neg_head_idx = torch.nonzero(~mask, as_tuple=True)[0]
    
    cont_loss = []
    for p_idx in pos_head_idx:
        pos_feats = a_feats[torch.nonzero(labels == p_idx, as_tuple=True)[0].long()]
        pos_sim = torch.exp(torch.cosine_similarity(a_heads[p_idx].unsqueeze(0), pos_feats)).mean()
        
        neg_feats = a_feats[torch.nonzero(labels != p_idx, as_tuple=True)[0].long()]
        neg_heads = a_heads[torch.arange(a_heads.shape[0]) != p_idx]
        
        neg_sim = torch.exp(torch.cosine_similarity(a_heads[p_idx].unsqueeze(0), neg_feats)).mean() + torch.exp(torch.cosine_similarity(a_heads[p_idx].unsqueeze(0), neg_heads)).mean()
        
        cont_loss.append((neg_sim/pos_sim).unsqueeze(0))
        

    cont_loss = torch.cat(cont_loss, dim=0)
    
    return cont_loss.mean()

from scipy.optimize import linear_sum_assignment as linear_assignment
def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_arr, jnd_arr = linear_assignment(w.max() - w)
    ind = np.array(list(zip(ind_arr, jnd_arr)))

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind

    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


#* ===============================================================

# def Attn_Contrastive_Loss(a_feats, labels, a_heads):
#     mask = torch.zeros(len(a_heads), dtype=torch.bool)
#     mask[torch.unique(labels)] = True

#     pos_head_idx = torch.nonzero(mask, as_tuple=True)[0]
#     neg_head_idx = torch.nonzero(~mask, as_tuple=True)[0]
    
#     cont_loss = []
#     for p_idx in pos_head_idx:
#         pos_feats = a_feats[torch.nonzero(labels == p_idx, as_tuple=True)[0].long()]
#         pos_sim = torch.exp(torch.cosine_similarity(a_heads[p_idx].unsqueeze(0), pos_feats)).mean()
        
#         neg_feats = a_feats[torch.nonzero(labels != p_idx, as_tuple=True)[0].long()]
#         neg_heads = a_heads[torch.arange(a_heads.shape[0]) != p_idx]
        
#         neg_sim = torch.exp(torch.cosine_similarity(a_heads[p_idx].unsqueeze(0), neg_feats)).mean() + torch.exp(torch.cosine_similarity(a_heads[p_idx].unsqueeze(0), neg_heads)).mean()
        
#         #* 0.5 --> Margin..
#         cont_loss.append((neg_sim-pos_sim-0.5).unsqueeze(0))

#     cont_loss = torch.cat(cont_loss, dim=0)
    
#     return cont_loss.mean()