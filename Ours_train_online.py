import argparse, os, copy, random, sys
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from functools import partial

from tqdm import *

import dataset, utils, losses, net
from dataset.get_data_utils import get_class_splits
from net.resnet import *
from net.vision_transformer import VisionTransformer, ViT_Dino, ViT_IN21K, DINOHead

from dataset.cub import subsample_classes, online_subsample_dataset

from Ours_utils.utils import extract_embedding, discover_acc

from net.Decoder import Head

def generate_dataset(dataset, idxs):
    dataset_ = copy.deepcopy(dataset)

    uq_mask = np.zeros(len(dataset_)).astype('bool')
    uq_mask[idxs] = True
    
    dataset_.uq_idxs = dataset.uq_idxs[uq_mask]
    dataset_.data = dataset.data[uq_mask]
    dataset_.ys = dataset.ys[uq_mask] 

    return dataset_

def generate_dataset_in(dataset, idxs):
    dataset_ = copy.deepcopy(dataset)

    uq_mask = np.zeros(len(dataset_)).astype('bool')
    uq_mask[idxs] = True
    
    dataset.imgs = dataset_.imgs[uq_mask]
    dataset.samples = dataset_.samples[uq_mask]
    dataset.targets = np.array(dataset_.targets[uq_mask]).astype(int)
    dataset.ys = dataset_.ys[uq_mask]
    dataset.uq_idxs = dataset_.uq_idxs[uq_mask]


    return dataset_

import pandas as pd

def merge_dataset(dataset_o, dataset_n):
    dataset_ = copy.deepcopy(dataset_o)
    old_uq = dataset_o.uq_idxs
    new_uq = dataset_n.uq_idxs
    
    merge_uq_idxs = list(old_uq) + list(new_uq)
    # dataset_.data = pd.concat([dataset_o.data, dataset_n.data])
    if args.dataset == 'cub':
        import pandas as pd
        dataset_.data = pd.concat([dataset_o.data, dataset_n.data])
    else:
        dataset_.data = np.concatenate([dataset_o.data, dataset_n.data])
    dataset_.uq_idxs = np.array(merge_uq_idxs)
    dataset_.ys = np.concatenate((dataset_o.ys, dataset_n.ys))
    
    return dataset_

def merge_dataset_in(dataset_o, dataset_n):
    dataset_ = copy.deepcopy(dataset_o)
    old_uq = dataset_o.uq_idxs
    new_uq = dataset_n.uq_idxs
    
    merge_uq_idxs = list(old_uq) + list(new_uq)
    # dataset_.data = pd.concat([dataset_o.data, dataset_n.data])
    dataset_.imgs = np.concatenate([dataset_o.imgs, dataset_n.imgs])
    dataset_.samples = np.concatenate([dataset_o.samples, dataset_n.samples])
    dataset_.targets = np.concatenate([dataset_o.targets, dataset_n.targets])
    
    dataset_.uq_idxs = np.array(merge_uq_idxs)
    dataset_.ys = np.concatenate((dataset_o.ys, dataset_n.ys))
    
    return dataset_

def set_seed(seed):
    if seed == -1:
        print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                     + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`')
    # export directory, training and val datasets, test datasets
    parser.add_argument('--LOG_DIR', default='./logs', help='Path to log folder')
    parser.add_argument('--dataset', default='cub', help='Training dataset, e.g. cub, cars, SOP, Inshop') # cub # mit # dog # air
    parser.add_argument('--embedding-size', default=768, type=int, dest='sz_embedding', help='Size of embedding that is appended to backbone model.')
    parser.add_argument('--batch-size', default=120, type=int, dest='sz_batch', help='Number of samples per batch.')  # 150
    parser.add_argument('--inc_batch-size', default=32, type=int, dest='inc_sz_batch', help='Number of samples per batch.')  # 150
    parser.add_argument('--base_epochs', default=60, type=int, dest='nb_epochs', help='Number of training epochs.')
    parser.add_argument('--inc_epochs', default=10, type=int, dest='inc_epochs', help='Number of training epochs.')

    parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU that is used for training.')

    parser.add_argument('--workers', default=2, type=int, dest='nb_workers', help='Number of workers for dataloader.')
    # parser.add_argument('--workers', default=4, type=int, dest='nb_workers', help='Number of workers for dataloader.')
    parser.add_argument('--model', default='resnet18', help='Model for training')  # resnet50 #resnet18  VIT
    parser.add_argument('--loss', default='Proxy_Anchor', help='Criterion for training') #Proxy_Anchor #Contrastive
    parser.add_argument('--optimizer', default='adamw', help='Optimizer setting')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate setting')  #1e-4
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay setting')
    parser.add_argument('--lr-decay-step', default=5, type=int, help='Learning decay step setting')  #
    parser.add_argument('--lr-decay-gamma', default=0.5, type=float, help='Learning decay gamma setting')
    parser.add_argument('--alpha', default=32, type=float, help='Scaling Parameter setting')
    parser.add_argument('--mrg', default=0.1, type=float, help='Margin parameter setting')
    parser.add_argument('--warm', default=5, type=int, help='Warmup training epochs')  # 1
    parser.add_argument('--bn-freeze', default=True, type=bool, help='Batch normalization parameter freeze')
    parser.add_argument('--l2-norm', default=True, type=bool, help='L2 normlization')
    parser.add_argument('--remark', default='', help='Any reamrk')

    parser.add_argument('--use_split_modlue', type=bool, default=True)
    parser.add_argument('--use_GM_clustering', action='store_true') # False

    parser.add_argument('--exp', type=str, default='0')
    
    ###=========================================
    parser.add_argument('--num_inc_sessions', default=1, type=int, help='The number of incremental sessions')  # 1
    parser.add_argument('--num_base_classes', default=160, type=int, help='The number of incremental sessions')  # 1
    # parser.add_argument('--vit_pretrained_dino', default='/data/pgh2874/Anytime_Novel_Discovery/CGCD/pretrained_weight/dino_ViT-B16_backbone.pth', help='Path to log folder')
    parser.add_argument('--vit_pretrained_dino', default='/data/pgh2874/Anytime_Novel_Discovery/CGCD/pretrained_weight/dino_vitbase16_pretrain_full_checkpoint.pth', help='Path to log folder')
    
    parser.add_argument('--seed', default=1, type=int, help='Experiment seed')  # 1
    parser.add_argument('--gat_n_layers', default=4, type=int, help='Num layers of GAT')  # 1
    parser.add_argument("--cont_param", default=1, type=float, help='Anytime Novel Discovery')
    
    # parser.add_argument("--cont_attn", action='store_true', help='Anytime Novel Discovery')
    parser.add_argument('--prompt_tuning_layers', nargs='+', type=int, help='<Required> prefix tuning layers')
    parser.add_argument('--prompt_leng', default=10, type=int, help='prompt length')  # 1
    
    parser.add_argument('--adapt_mlp_layers', nargs='+', type=int, help='<Required> Adapt MLP layers')
    parser.add_argument('--adapter_layers', nargs='+', type=int, help='<Required> Adapt MLP layers')
    parser.add_argument('--lora_layers', nargs='+', type=int, help='<Required> Adapt MLP layers')
    parser.add_argument('--lora_prefix', action='store_true')
    
    parser.add_argument('--n_replay', default=5, type=int, help='prompt length')  # 1
    
    parser.add_argument('--use_EC', action='store_true')
    parser.add_argument('--use_VFA', action='store_true')
    parser.add_argument('--energy_hp', type=float, default=1.)
    parser.add_argument('--prop_samples', type=float, default=0.8)
    

    ####
    args = parser.parse_args()
    set_seed(args.seed)
    
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
    
    args = get_class_splits(args)
    
    score_summary = {
        'pseudo_acc_avg': [],
        'pseudo_acc_labeled': [],
        'pseudo_acc_unlabeled': [],
        
        'Labeled_discover_li':[],
        'Unlabeled_discover_li':[],
        'All_discover_li':[],
        
        'confidence_labeled': [],
        'confidence_unlabeled': []
    }
    report_dict = {
        'grad_norm': []
    }
    report_path = f'./report_dict/Latents/{args.exp}'
    os.makedirs(report_path, exist_ok=True)
    grad_path = f'./report_dict/grads/{args.exp}'
    os.makedirs(grad_path, exist_ok=True)
    ####
    pth_rst = './Ours/' + args.dataset
    os.makedirs(pth_rst, exist_ok=True)
    pth_rst_exp = pth_rst + '/' + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
    os.makedirs(pth_rst_exp, exist_ok=True)

    ####
    # pth_dataset = '../datasets'
    pth_dataset = '/local_datasets/khpark/AND'
    if args.dataset == 'cub':
        pth_dataset += '/CUB_200_2011'
    elif args.dataset == 'cifar100':
        pth_dataset += '/cifar-100-python'
    elif args.dataset == 'air':
        pth_dataset += '/fgvc-aircraft-2013b'
    elif args.dataset == 'imagenet_100':
        pth_dataset = "/local_datasets/imagenet"

    dt_train_labelled, dt_train_inc_unlabelled, dt_test_labelled, dt_test_inc_unlabelled, whole_test_set, whole_train_set = dataset.load(name=args.dataset, root=pth_dataset, args=args, train_transform=dataset.utils.make_transform(is_train=True), test_transform=dataset.utils.make_transform(is_train=False))
    
    dl_train_labelled = torch.utils.data.DataLoader(dt_train_labelled, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)
    # dl_train_labelled.dataset.twice_trsf = weak_trsf_pairwise()
    #* nb_classes = dset_tr_0.nb_classes()
    nb_classes = args.num_base_classes
    #todo ============= DATA Prepared =============
    #! Mehotd + Main Code 수정!!!!
    #### Backbone Model
    #todo bn_freeze default True
    if args.model.find('resnet18') > -1:
        model = Resnet18(embedding_size=args.sz_embedding, pretrained=False, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
    elif args.model.find('resnet50') > -1:
        model = Resnet50(embedding_size=args.sz_embedding, pretrained=False, is_norm=args.l2_norm, bn_freeze=args.bn_freeze, num_classes=None)
    elif args.model.find('VIT') > -1:
        model = ViT_Dino(args)
        model.load_state_dict(torch.load(args.vit_pretrained_dino)['teacher'], strict=False)
        model.backbone.embedding = nn.Linear(model.backbone.embed_dim, args.sz_embedding)
        # model = ViT_IN21K(args)
        # model.backbone.embedding = nn.Identity()
        
        model.head= nn.Identity()
        
        print("Adopt ViT pretrained on ImageNet via DINO!!")
    else:
        print('?')
        sys.exit()


    model = model.cuda()
    header = Head(args.sz_embedding,nb_classes)
    header = header.cuda()
    
    
    param_groups = [
        # {'params': list(set(model_now.parameters()).difference(set(list(model_now.backbone.embedding.parameters())+list(model_now.backbone.etf_embedding.parameters()))))},
        {'params': list(set(model.parameters()).difference(set(list(model.backbone.embedding.parameters()))))},
        {'params': model.backbone.embedding.parameters(), 'lr': float(args.lr) * 1},
        # {'params': model.backbone.etf_embedding.parameters(), 'lr': float(args.lr) * 1},
        
        # {'params': attn_head.parameters(), 'lr': float(args.lr)},
        # {'params': attn_head.parameters(), 'lr': 5e-3},
        ]

    
    param_groups.append({'params': header.parameters(), 'lr': float(args.lr)})
    
    #### Optimizer
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.nb_epochs, eta_min = 1e-6)

    # for p in model.parameters():
    #     p.requires_grad=True
    
    print('Argparser Namespace: {}'.format(vars(args)))
    
    model_parameters = sum(p.numel() for p in model.parameters())
    model_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad ==True)
    header_parameters = sum(p.numel() for p in header.parameters())
    # proj_learnable_parameters = sum(p.numel() for p in projection_head.parameters() if p.requires_grad ==True)
    
    print('ViT Parameters: {}'.format(model_parameters))
    print('Learnable ViT Parameters: {}'.format(model_learnable_parameters))
    print("Header:", header_parameters)
    
    print('Training for {} epochs'.format(args.nb_epochs))
    losses_list = []
    # best_recall = [0]
    best_recall = 0.
    best_epoch = 0

    #### Load checkpoint.
    # dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', transform=dataset.utils.make_transform(is_train=False))
    dl_test_labelled = torch.utils.data.DataLoader(dt_test_labelled, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    for epoch in range(0, args.nb_epochs):
        model.train()
        header.train()
        losses_per_epoch = []
        #### Warmup: Train only new params, helps stabilize learning.
        if args.warm > 0:
            # unfreeze_model_param = list(model.backbone.embedding.parameters()) + list(model_D.parameters())
            unfreeze_model_param = list(model.backbone.embedding.parameters())

            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

        total, correct = 0, 0
        pbar = tqdm(enumerate(dl_train_labelled), disable=True)
        for batch_idx, (x, y, z) in pbar:
            #todo input x --> x, x_bar로 수정함 (Augmented View를 통해 Contrastive Loss 활용하자)
            x = x.cuda()
            y = y.cuda()
            
            #todo randomly chosen cutmix or mixup
            # mixed_x, mixed_target_dist, lam = cutmix_or_mixup(x, y, nb_classes)
            
            feats = model(x)    #* 2B, dim
            # known_feats, unknown_feats = feats[:x.shape[0]], feats[x.shape[0]:]
            # known_feats, mixed_feats = feats[:x.shape[0]], feats[x.shape[0]:]
            
            logits = header(feats)  #* 2B, 2, Num CLS
            ce_loss = F.cross_entropy(logits, y)
            
            # mixed_logits = header(mixed_feats)  #* 2B, 2, Num CLS
            # mixed_ce_loss = F.cross_entropy(mixed_logits, mixed_target_dist)
            
            opt.zero_grad()
            
            # loss = 0.5*(ce_loss_pos + ce_loss_neg) + 0.5*(kki_loss + kui_loss) + mixed_ce_loss
            # loss = ce_loss + mixed_ce_loss + 10.*d_loss
            loss = ce_loss
            
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            torch.nn.utils.clip_grad_value_(header.parameters(), 10)
            # torch.nn.utils.clip_grad_value_(model_D.parameters(), 10)
            

            losses_per_epoch.append(loss.data.cpu().numpy())
            opt.step()

            pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} CE_Loss:{:.4f}'\
                .format(epoch, batch_idx + 1, len(dl_train_labelled), 100. * batch_idx / len(dl_train_labelled), loss.item(), ce_loss.item()))

        losses_list.append(np.mean(losses_per_epoch))
        scheduler.step()
        
        if (epoch >= 0):
            with torch.no_grad():
                print('Evaluating..')
                model.eval()
                header.eval()
                all_preds = []
                targets = np.array([])
                for batch_idx, (x, y, _) in enumerate(tqdm(dl_test_labelled, disable=True)):
                    x, y = x.cuda(), y.cuda()
                    feats = model(x)
                    logits = header(feats)
                    _, preds = logits.max(dim=1)
                    all_preds.append(preds.cpu().numpy())
                    targets = np.append(targets, y.cpu().numpy())
                
                all_preds = np.concatenate(all_preds)
                acc_0, _ = utils._hungarian_match_(all_preds, np.array(dl_test_labelled.dataset.ys[dl_test_labelled.dataset.uq_idxs]))
                # acc_0 = cluster_acc(targets.astype(int), all_preds.astype(int))
                
                # acc_0 = cluster_acc(targets.astype(int), preds.astype(int))
                print('Test acc {:.4f}'.format(acc_0))

            #### Best model save
            if best_recall < acc_0:
                best_recall = acc_0
                best_epoch = epoch
                torch.save({'model_state_dict': model.state_dict(), 'header_state_dict':header.state_dict()}, '{}/{}_{}_best_step_0.pth'.format(pth_rst_exp, args.dataset, args.model))

    ####
    print('==> Resuming from checkpoint..')
    pth_pth = pth_rst_exp + '/' + '{}_{}_best_step_{}.pth'.format(args.dataset, args.model, 0)

    checkpoint = torch.load(pth_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    header.load_state_dict(checkpoint['header_state_dict'])
    # model_D.load_state_dict(checkpoint['model_D_state_dict'])
    # attn_head.load_state_dict(checkpoint['attn_head_state_dict'])

    model = model.cuda()
    header = header.cuda()
    # model_D = model_D.cuda()
    # attn_head = attn_head.cuda()
    ####
    print('==> Init. Evaluation..')
    with torch.no_grad():
        model.eval()
        header.eval()
        all_preds = []
        targets = np.array([])
        for batch_idx, (x, y, _) in enumerate(tqdm(dl_test_labelled, disable=True)):
            x, y = x.cuda(), y.cuda()
            feats = model(x)
            logits = header(feats)
            _, preds = logits.max(dim=1)
            all_preds.append(preds.cpu().numpy())
            targets = np.append(targets, y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        acc_0, _ = utils._hungarian_match_(all_preds, np.array(dl_test_labelled.dataset.ys[dl_test_labelled.dataset.uq_idxs]))
        
        # acc_0 = cluster_acc(targets.astype(int), all_preds.astype(int))
    print('[Base Session] After Pretrain (Base Session) Acc: {:.4f}'.format(acc_0))     
    score_summary['Base'] = acc_0
    
    dlod_tr_prv = dl_train_labelled
    nb_classes_prv = nb_classes
    
    list_base = np.arange(nb_classes)
    online_seen_classes = [i for i in list_base]
    #todo Incrementals..==============================================================
    list_old_inc = np.array([])
    energy_store = np.array([])
    #? dt_train_labelled, dt_train_inc_unlabelled, dt_test_labelled, dt_test_inc_unlabelled
    for sess_idx, (inc_tr_dataset, inc_ev_dataset) in enumerate(zip(dt_train_inc_unlabelled, dt_test_inc_unlabelled)):
        #? dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=False))
        #? dset_ev_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_ev_now_md, transform=dataset.utils.make_transform(is_train=False))
        inc_tr_dataset.transform = dataset.utils.make_transform(is_train=False)
        inc_ev_dataset.transform= dataset.utils.make_transform(is_train=False)
        
        dlod_tr_now = torch.utils.data.DataLoader(copy.deepcopy(inc_tr_dataset), batch_size=args.inc_sz_batch, shuffle=True, num_workers=args.nb_workers)
        #todo=====================
        list_new_inc = inc_tr_dataset.new_inc
        #todo=====================
        
        inc_pth_rst_exp = os.path.join(pth_rst_exp,f'Inc_{sess_idx}')
        os.makedirs(inc_pth_rst_exp, exist_ok=True)
        
        model_now = copy.deepcopy(model)
        header_now = copy.deepcopy(header)
        
        for p in model.parameters():
            p.requires_grad=False
        
        for p in header.parameters():
            p.requires_grad= False
        
        for p in model_now.parameters():
            p.requires_grad=False
        
        # if args.historical_prompt:
        #     model_now.backbone.prepare_historical_prompt_tuning(n_length=args.prompt_leng)
        
        if args.prompt_tuning_layers is not None:
            model_now.backbone.prepare_prompt(tuning_layers= args.prompt_tuning_layers, n_length=args.prompt_leng)
        
        if args.lora_layers is not None:
            if args.lora_prefix:
                model_now.backbone.prepare_LoRA(lora_layers=args.lora_layers, p_leng=args.prompt_leng, prefix=args.lora_prefix)
            else:
                model_now.backbone.prepare_LoRA(lora_layers=args.lora_layers)
        
        if args.adapt_mlp_layers is not None:
            model_now.backbone.prepare_adapt_MLP(adapt_mlp_layers= args.adapt_mlp_layers)
        
        model_now = model_now.cuda()
        header_now = header_now.cuda()
        
        model_parameters = sum(p.numel() for p in model_now.parameters())
        
        model_learnable_parameters = sum(p.numel() for p in model_now.parameters() if p.requires_grad)
        print('Total Parameters: {}'.format(model_parameters))
        print('Learnable Parameters: {}'.format(model_learnable_parameters))
        epoch = 0
        nb_classes_old = nb_classes_prv
        nb_classes_k_online = 0
        nb_classes_now = 0
        
        from splitNet import SplitModlue
        split_module = SplitModlue(save_path=inc_pth_rst_exp,sz_feature=args.sz_embedding)
        pbar = enumerate(dlod_tr_now)
        for batch_idx, (on_x, on_y, on_z) in pbar:
            for on_cls in torch.unique(on_y):
                if on_cls not in online_seen_classes:
                    online_seen_classes.append(on_cls.item())
            print("\nOnline Seen Classes:", online_seen_classes)
            #todo=====================================================================
            
            #!todo  서버 다운으로 Subsample_dataset으로 수정이후 test 못해봄 
            #! dummy_dset = utils.dummy_dataset(copy.deepcopy(whole_train_set), batch_uq_idxs=on_z, transform=dataset.utils.make_transform(is_train=False))
            
            # dummy_dset = online_subsample_dataset(copy.deepcopy(whole_train_set), idxs=on_z)
            if args.dataset=='cub':
                dummy_dset = online_subsample_dataset(copy.deepcopy(whole_train_set), idxs=on_z)
            elif args.dataset == 'imagenet_100':
                from dataset.imagenet import online_subsample_dataset as in_online_subsample_dataset
                dummy_dset = in_online_subsample_dataset(copy.deepcopy(whole_train_set), idxs=on_z)
            else:
                from dataset.air import online_subsample_dataset as air_online_subsample_dataset
                dummy_dset = air_online_subsample_dataset(copy.deepcopy(whole_train_set), idxs=on_z)
            
            dummy_dset.transform = dataset.utils.make_transform(is_train=False)
            dlod_dummy_tr_now = torch.utils.data.DataLoader(copy.deepcopy(dummy_dset), batch_size=len(on_z), shuffle=False, num_workers=args.nb_workers)
            #todo 1. Split
            print(f'[Incremental Session:{sess_idx}] ==> Split Labeled and Unlabeled')
            thres = 0.
            with torch.no_grad():
                prev_feats, prev_labels, _ = extract_embedding(model, dlod_dummy_tr_now)
                # prev_logits = F.linear(a_prev_feats, a_prev_heads)
                prev_logits = header(prev_feats)
                energy_labeled  = -torch.logsumexp(prev_logits, dim=1).unsqueeze(1)
                
                gm = GaussianMixture(n_components=2, max_iter=1000, tol=1e-4, init_params='kmeans', random_state=args.seed).fit(energy_labeled.cpu().numpy()) 
                preds_lb_n = gm.predict_proba(energy_labeled.cpu().numpy())
                
                pred = preds_lb_n.argmax(1)
                clus_a = np.where(0 == pred)[0]
                clus_b = np.where(1 == pred)[0]
                if (-1*energy_labeled[clus_a]).mean() < (-1*energy_labeled[clus_b]).mean(): #* cluster B의 Energy가 더 큰 경우
                    idx_o = clus_b
                    idx_n = clus_a
                else: #* cluster A의 Energy가 더 큰 경우
                    idx_o = clus_a
                    idx_n = clus_b
                print("#"*50)
                print("Splited Labeled Idx:", len(idx_o), "Energy:", energy_labeled[idx_o].mean().item())
                print("Splited Unlabeled Idx:", len(idx_n), "Energy:", energy_labeled[idx_n].mean().item())
            
            discover_score = discover_acc(idx_o, idx_n, prev_labels, nb_classes)
            
            print(f'[Incremental Session:{sess_idx}] ==> Fine. Split old and new..')
            
            score_summary['Labeled_discover_li'].append(discover_score['old_detect_acc'])
            score_summary['Unlabeled_discover_li'].append(discover_score['new_detect_acc'])
            score_summary['All_discover_li'].append(discover_score['all_detect_acc'])
            
            print("#"*50)
            print("Splited Labeled Idx:", len(idx_o))
            print("Splited Unlabeled Idx:", len(idx_n))
            print("#"*50)
            print()
            #? generate_dataset(dataset, idxs)
            # print("Dataset_Train_OLD")
            if args.dataset == 'imagenet_100':
                # print('dummy_dset:',dummy_dset.uq_idxs, len(dummy_dset.uq_idxs))
                # print('idx_o',idx_o, len(idx_o))
                # print('idx_n',idx_n, len(idx_n))
                dset_tr_o = generate_dataset_in(copy.deepcopy(dummy_dset), idx_o)
                # print('dummy_dset:',dummy_dset.uq_idxs, len(dummy_dset.uq_idxs))
                dset_tr_n = generate_dataset_in(copy.deepcopy(dummy_dset), idx_n)
                # whole_train_set
            else:
                dset_tr_o = generate_dataset(dummy_dset, idx_o)
                dset_tr_n = generate_dataset(dummy_dset, idx_n)
            dlod_tr_o = torch.utils.data.DataLoader(dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
            dlod_tr_n = torch.utils.data.DataLoader(dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
            ####
            print(f'[Incremental Session:{sess_idx}] ==> Replace old labels..')
            if len(dset_tr_o) > 0:
                with torch.no_grad():
                    # feats, _ = utils.evaluate_cos_(model, dlod_tr_o)
                    feats, _, _ = extract_embedding(model, dlod_tr_o)
                    # cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(header.head.weight))
                    prev_logits = header(feats)
                    #! preds_val, preds_lb = torch.max(prev_logits, dim=1)
                    preds_val, preds_lb = torch.max(F.softmax(prev_logits, dim=1), dim=1)
                    score_summary['confidence_labeled'].append(preds_val.cpu())
                    
                    preds_lb_o = preds_lb.detach().cpu().numpy()

            ####
            print(f'[Incremental Session:{sess_idx}] ==> Clustering splitted new and replace new labels..')
            if len(dset_tr_n) > 0:
                if batch_idx > 0:
                    with torch.no_grad():
                        feats, _, _ = extract_embedding(model_now, dlod_tr_n)
                        logits = F.linear(feats, header_now.head.weight[nb_classes:], header_now.head.bias[nb_classes:])
                        energy_unlabeled  = -torch.logsumexp(logits, dim=1).unsqueeze(1)
                    
                        gm = GaussianMixture(n_components=2, max_iter=1000, tol=1e-4, init_params='kmeans', random_state=args.seed).fit(energy_unlabeled.cpu().numpy()) 
                        preds_unlabeled = gm.predict_proba(energy_unlabeled.cpu().numpy())
                    
                    pred = preds_unlabeled.argmax(1)
                    clus_a = np.where(0 == pred)[0]
                    clus_b = np.where(1 == pred)[0]
                    if (-1*energy_unlabeled[clus_a]).mean() < (-1*energy_unlabeled[clus_b]).mean(): #* cluster B의 Negative Energy가 더 큰 경우
                        idx_seen = clus_b
                        idx_unseen = clus_a
                    else: #* cluster A의 Negative Energy가 더 큰 경우
                        idx_seen = clus_a
                        idx_unseen = clus_b
                    print("#"*50)
                    print("Unlabeled Seen Idx:", len(idx_seen), "Energy:", energy_unlabeled[idx_seen].mean().item())
                    print("Unlabeled Unseen Idx:", len(idx_unseen), "Energy:", energy_unlabeled[idx_unseen].mean().item())
                    print("#"*50)
                    print()

                    #todo Pseudo Labeling seen categories
                    p_seen_logit = F.linear(feats[idx_seen], header_now.head.weight[nb_classes:], header_now.head.bias[nb_classes:])
                    #! seen_pred_val, seen_pred_lb = torch.max(p_seen_logit, dim=1)
                    seen_pred_val, seen_pred_lb = torch.max(F.softmax(p_seen_logit, dim=1), dim=1)
                    score_summary['confidence_unlabeled'].append(seen_pred_val.cpu())
                    # 'confidence_labeled': [],
                    # 'confidence_unlabeled': []
                    
                    
                    pseudo_seen = seen_pred_lb.detach().cpu().numpy()
                else:   #* First Online Batch
                    with torch.no_grad():
                        feats, _, _ = extract_embedding(model, dlod_tr_n)
                    pseudo_seen = None
                    idx_unseen = np.arange(feats.shape[0])
                    
                #todo ========================================
                #todo Unseen Proliferation for enhanced noise Labeling
                if args.use_VFA:
                    unseen_feats = feats[idx_unseen].cpu()
                    replayed_feats = []
                    unseen_std = unseen_feats.std(dim=0)
                    for uns_f in unseen_feats:
                        tmp_f = [torch.normal(uns_f, unseen_std) for _ in range(args.n_replay)]
                        replayed_feats.append(torch.stack(tmp_f))
                    replayed_feats = torch.cat(replayed_feats, dim=0)
                    
                    proliferated_feats = torch.cat([unseen_feats, replayed_feats], dim=0) 
                else:
                    proliferated_feats = feats[idx_unseen].cpu()
                
                #todo ========================================
                
                # clst_a = AffinityPropagation(random_state=args.seed).fit(feats[idx_unseen].cpu().numpy()) # 0.75
                clst_a = AffinityPropagation(random_state=args.seed).fit(proliferated_feats.cpu().numpy()) # 0.75
                p, c = np.unique(clst_a.labels_, return_counts=True)
                nb_classes_k = len(p)
                print("nb_classes_k:", nb_classes_k)
                pseudo_unseen = clst_a.labels_

                ####
                # if args.use_GM_clustering:
                gm = GaussianMixture(n_components=nb_classes_k, max_iter=1000, tol=1e-4, init_params='kmeans', random_state=args.seed).fit(proliferated_feats.cpu().numpy()) 
                # gm = GaussianMixture(n_components=nb_classes_k, max_iter=1000, tol=1e-4, init_params='kmeans', random_state=args.seed).fit(feats[idx_unseen].cpu().numpy()) 
                pseudo_unseen = gm.predict(feats[idx_unseen].cpu().numpy())
            else:
                nb_classes_k=0

            
            #todo =============================== Need to Analysis
            if args.dataset == 'imagenet_100':
                dset_tr_o = generate_dataset_in(copy.deepcopy(dummy_dset), idx_o)
                dset_tr_n = generate_dataset_in(copy.deepcopy(dummy_dset), idx_n)
            else:
                dset_tr_o = generate_dataset(dummy_dset, idx_o)
                dset_tr_n = generate_dataset(dummy_dset, idx_n)
            
            # dset_tr_o = generate_dataset(dummy_dset, idx_o)
            # dset_tr_n = generate_dataset(dummy_dset, idx_n)
            if len(dset_tr_o) > 0:
                dset_tr_o.ys = preds_lb_o.tolist()
            if len(dset_tr_n) > 0:
                #* Seen Unseen 구별 필요!!
                # pseudo_seen, idx_seen
                # pseudo_unseen, idx_unseen
                
                preds_lb_new = np.array([0 for _ in range (len(dset_tr_n))])
                if pseudo_seen is not None:
                    preds_lb_new[idx_seen] = pseudo_seen + nb_classes_prv
                    preds_lb_new[idx_unseen] = pseudo_unseen.astype(int) + nb_classes_now
                else:   #* First Batch (All the samples are unlabeled)
                    preds_lb_new[idx_unseen] = pseudo_unseen.astype(int) + nb_classes_prv
                
                # print('preds_lb_new',preds_lb_new)
                # print('nb_classes_now',nb_classes_now)
                
                if batch_idx == 0:
                    dset_tr_n.ys = (preds_lb_new).tolist()
                else:
                    dset_tr_n.ys = (preds_lb_new).tolist()
                # print('dset_tr_n.ys:',dset_tr_n.ys)
                
            #* dset_tr_n.ys = (preds_lb_n).tolist()
            if args.dataset == 'imagenet_100':
                dset_tr_now_m = merge_dataset_in(dset_tr_o, dset_tr_n)
            else:
                dset_tr_now_m = merge_dataset(dset_tr_o, dset_tr_n)
            dset_tr_now_m.transform = dataset.utils.make_transform(is_train=True)
            dlod_tr_now_m = torch.utils.data.DataLoader(dset_tr_now_m, batch_size=args.sz_batch*2, shuffle=True, num_workers=args.nb_workers)
            
            dset_tr_now_m.binary = True
            dset_tr_now_m.ablation= True
            ####
            print(f'[Incremental Session:{sess_idx}] ==> Training splitted new..')
            nb_classes_old = nb_classes_now
            nb_classes_now = nb_classes_prv + nb_classes_k_online + nb_classes_k
            #todo 2. Proxy Update
            # if batch_idx == 0:
            #     criterion_pa_now = losses.Proxy_Anchor(nb_classes=nb_classes_now, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).cuda()
            #     criterion_pa_now.proxies.data[:nb_classes_prv] = criterion_pa.proxies.data
            # else:
            #     criterion_pa_now = losses.Proxy_Anchor(nb_classes=nb_classes_now, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).cuda()
            #     criterion_pa_now.proxies.data[:nb_classes_old] = criterion_pa_old.proxies.data
            
            header_now.update(nb_classes_now=nb_classes_now)
            header_now = header_now.cuda()
            #todo ========================PET Position========================
            
            #todo ============================================================
            
            print("#"*50)
            print("Base Head Nodes")
            print(header.head.weight.shape)
            print("Updated Head Nodes")
            print(header_now.head.weight.shape)
            print("#"*50)
            
            #todo 3. Optimizer Update
            if batch_idx>0:
                #todo online optimizer update...
                opt_dict = copy.deepcopy(opt.state_dict())
                fc_params = opt_dict['param_groups'][-1]['params']
                
                if len(opt_dict['state']) > 0:
                    fc_weight_state = opt_dict['state'][fc_params[0]]
                    fc_bias_state = opt_dict['state'][fc_params[1]]
                    
                for param in opt.param_groups[-1]['params']:
                    if param in opt.state.keys():
                        del opt.state[param]
                #todo ===============================================
                del opt.param_groups[-1]
                opt.add_param_group({'params' : header_now.parameters(), 'lr':float(args.lr) * 1})
                
                if len(opt_dict['state']) > 0:
                    # for item in opt.param_groups[-1]:
                    #     print(item)
                    fc_weight = opt.param_groups[-1]['params'][0]
                    fc_bias = opt.param_groups[-1]['params'][1]
                    
                    # print("fc_weight_state['exp_avg']:", fc_weight_state['exp_avg'].shape)
                    # print("fc_bias_state['exp_avg']:", fc_bias_state['exp_avg'].shape)
                    
                    opt.state[fc_weight]['step'] = fc_weight_state['step']
                    opt.state[fc_weight]['exp_avg'] = torch.cat([fc_weight_state['exp_avg'], torch.zeros((nb_classes_k, fc_weight_state['exp_avg'].size(dim=1))).cuda()], dim=0)
                    opt.state[fc_weight]['exp_avg_sq'] = torch.cat([fc_weight_state['exp_avg_sq'], torch.zeros((nb_classes_k, fc_weight_state['exp_avg_sq'].size(dim=1))).cuda()], dim=0)
                    
                    opt.state[fc_bias]['step'] = fc_bias_state['step']
                    opt.state[fc_bias]['exp_avg'] = torch.cat([fc_bias_state['exp_avg'], torch.zeros(nb_classes_k, device='cuda')], dim=0)
                    opt.state[fc_bias]['exp_avg_sq'] = torch.cat([fc_bias_state['exp_avg_sq'], torch.zeros(nb_classes_k, device='cuda')], dim=0)
                    
            else:
                param_groups = [
                {'params': list(set(model_now.parameters()).difference(set(list(model_now.backbone.embedding.parameters()))))},
                {'params': model_now.backbone.embedding.parameters(), 'lr': float(args.lr) * 1},
                # {'params': model_D_now.parameters(), 'lr':float(args.lr)*100},
                
                # {'params': model_now.backbone.etf_embedding.parameters() if args.gpu_id != -1 else model_now.module.embedding.parameters(), 'lr': float(args.lr) * 1},
                
                # {'params': attn_head.parameters(), 'lr': float(args.lr) * 1},
                ]
                
                
                opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, betas=(0.9, 0.999))
                opt.add_param_group({'params' : header_now.parameters(), 'lr':float(args.lr)})
                
            #todo=====================================================================
            
            for on_x, on_y, gt_y, on_z in dlod_tr_now_m:
                #* 원래는 Loop 밖에 있었음
                model_now.train()
                header_now.train()
                # print('on_x:', on_x.shape)
                # print('on_y:', on_y, on_y.shape)
                
                on_x = on_x.cuda()
                on_y = on_y.cuda()
                
                pseudo_acc_avg, _ = utils._hungarian_match_(np.array(on_y.cpu()), np.array(gt_y.cpu()))
                score_summary['pseudo_acc_avg'].append(pseudo_acc_avg)
                
                lab_mask = gt_y <= nb_classes
                pseudo_acc_labeled, _ = utils._hungarian_match_(np.array(on_y[lab_mask].cpu()), np.array(gt_y[lab_mask].cpu()))
                score_summary['pseudo_acc_labeled'].append(pseudo_acc_labeled)
                
                pseudo_acc_unlabeled, _ = utils._hungarian_match_(np.array(on_y[~lab_mask].cpu()), np.array(gt_y[~lab_mask].cpu()))
                score_summary['pseudo_acc_unlabeled'].append(pseudo_acc_unlabeled)
                
                
                
                y_n = torch.where(on_y > nb_classes-1, 1, 0)
                y_o = on_y.size(0) - y_n.sum()
                
                y_o_msk = torch.nonzero(y_n)
                y_n_msk = torch.nonzero(1.-y_n)
                
                print('y_labeled: {} / y_unlabeled: {}'.format(y_o, y_n.sum().item()))
                for epoch in range(0, args.inc_epochs):
                    
                    # feats, prompt_feats = model_now(on_x, return_all_patches=True)
                    feats = model_now(on_x)
                    
                    if y_n_msk.size(0) > 0:
                        unlabeled_feats = feats[y_n_msk[0]]
                        
                        n_lab_logits = F.linear(unlabeled_feats, header_now.head.weight[:nb_classes], header_now.head.bias[:nb_classes])
                        energy_n_lab  = -torch.logsumexp(n_lab_logits, dim=1).unsqueeze(1)
                        
                        n_unlab_logits = F.linear(unlabeled_feats, header_now.head.weight[nb_classes:], header_now.head.bias[nb_classes:])
                        energy_n_unlab  = -torch.logsumexp(n_unlab_logits, dim=1).unsqueeze(1)
                        
                        unlabeled_energy_loss = torch.log(1+(energy_n_unlab/energy_n_lab)).mean()
                        
                    else:
                        unlabeled_energy_loss = torch.zeros(1, device='cuda')
                        # known_ent = torch.zeros(1, device='cuda')
                    
                    if args.use_EC:
                        energy_loss =  unlabeled_energy_loss
                    else:
                        energy_loss =  torch.zeros(1, device='cuda')
                    
                    logits = header_now(feats)
                    
                    ce_loss = F.cross_entropy(logits, on_y)
                    
                    loss = ce_loss + args.energy_hp * energy_loss

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                    print('[Inc Train] [{}/{} ({:.0f}%)] Epoch: {} Loss: {:.4f} CE_Loss: {:.4f} Energy_Loss: {:.4f}'\
                        .format(batch_idx + 1, len(dlod_tr_now), 100. * (batch_idx+1) / len(dlod_tr_now), epoch, loss.item(), ce_loss.item(), energy_loss.item()))
                    # print('[Inc Train] [{}/{} ({:.0f}%)] Epoch: {} Sup Con Loss: {:.4f} KD Loss: {:.4f}'\
                    #     .format(batch_idx + 1, len(dlod_tr_now), 100. * (batch_idx+1) / len(dlod_tr_now), epoch, sup_con_loss.item(), loss_kd.item()))

            # scheduler.step()
            nb_classes_k_online += nb_classes_k
            nb_classes_old = nb_classes_now
            # head_nodes_old = copy.deepcopy(head_nodes_now)
            #! Anytime Inference is needed..
            #! Including subset of Black Category
            #! Including Seen Classes..
            #!=============================
            print()
            print("Online Seen Classes")
            print(online_seen_classes, len(online_seen_classes))
            if batch_idx % 1 == 0:
                # anytime_eval_set = subsample_classes(copy.deepcopy(whole_test_set), include_classes=online_seen_classes)
                if args.dataset =='cub':
                    anytime_eval_set = subsample_classes(copy.deepcopy(whole_test_set), include_classes=online_seen_classes)
                elif args.dataset =='imagenet_100':
                    from dataset.imagenet import subsample_classes as in_subsample_classes
                    anytime_eval_set = in_subsample_classes(copy.deepcopy(whole_test_set), include_classes=online_seen_classes)
                else:
                    from dataset.air import subsample_classes as air_subsample_classes
                    anytime_eval_set = air_subsample_classes(copy.deepcopy(whole_test_set), include_classes=online_seen_classes)
                
                dlod_ev_now = torch.utils.data.DataLoader(anytime_eval_set, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
                print(f'[Incremental Session:{sess_idx}] ==> Any Time Inference..')
                model_now.eval()
                header_now.eval()
                with torch.no_grad():
                    # tmp_label = np.array([])
                    # for img, target, _ in dlod_ev_now:
                    #     tmp_label = np.concatenate((tmp_label, target))
                    preds_lb = np.array([])
                    targets = np.array([])
                    for batch_idx, (x,y,_) in enumerate(dlod_ev_now):
                        x, y = x.cuda(), y.cuda()
                        feats = model_now(x)
                        # feats = F.normalize(feats, dim=-1)
                        logits = header_now(feats)
                        _, pred = logits.max(dim=1)
                        # _, pred = F.linear(feats, F.normalize(head_nodes_now, dim=-1)).max(1)
                        # logits = head_nodes_now(feats)
                        
                        # _, pred = logits.max(1)
                        targets = np.append(targets, y.cpu().numpy())
                        preds_lb = np.append(preds_lb, pred.cpu().numpy())
                        
                    y = np.array(dlod_ev_now.dataset.ys[dlod_ev_now.dataset.uq_idxs])
                    proj_all_new = utils.cluster_pred_2_gt(preds_lb.astype(int), y.astype(int))
                    pacc_fun_all_new = partial(utils.pred_2_gt_proj_acc, proj_all_new)
                    acc_a = pacc_fun_all_new(y.astype(int), preds_lb.astype(int))

                    old_selected_mask = (y < nb_classes)    #* Base Session
                    acc_o = pacc_fun_all_new(y[old_selected_mask].astype(int), preds_lb[old_selected_mask].astype(int))
                    #todo =============================================================                
                    inc_new_selected_mask = np.zeros(len(y)).astype('bool')
                    inc_new_classes = np.setdiff1d(online_seen_classes, list_base)  #* white novel categories
                    print("inc_new_classes",inc_new_classes, len(inc_new_classes))
                    for new_cls in inc_new_classes:
                        inc_new_selected_mask[np.where(y == new_cls)[0]] = True
                    if sum(inc_new_selected_mask) > 0:
                        acc_inc_new = pacc_fun_all_new(y[inc_new_selected_mask].astype(int), preds_lb[inc_new_selected_mask].astype(int))
                    else:
                        acc_inc_new = 0.
                    
                    inc_old_mask = ~(old_selected_mask + inc_new_selected_mask)
                    if sum(inc_old_mask) > 0:
                        acc_inc_old = pacc_fun_all_new(y[inc_old_mask].astype(int), preds_lb[inc_old_mask].astype(int))
                    else:
                        acc_inc_old = 0.
                    print("="*60)
                    print('[Anytime Inference] All:{:.4f} / Base:{:.4f} / Inc_Old:{:.4f} / Inc_New:{:.4f}'.format(acc_a, acc_o, acc_inc_old, acc_inc_new))
                    print("="*60)
                    print()

        #todo Extract indirect knowledge of black categories
        dl_whole_test_set = torch.utils.data.DataLoader(copy.deepcopy(whole_test_set), batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
        list_base = np.arange(nb_classes)
        list_new_inc = dl_whole_test_set.dataset.new_inc
        
        list_seen = np.arange(len(args.train_classes)+len(args.unlabeled_classes))
        inc_new_classes = np.setdiff1d(list_new_inc, list_base)
        
        model_now.eval()
        header_now.eval()
        with torch.no_grad():
            preds_lb = np.array([])
            targets = np.array([])
            for batch_idx, (x,y,_) in enumerate(tqdm(dl_whole_test_set, disable=True)):
                x, y = x.cuda(), y.cuda()
                feats = model_now(x)
                logits = header_now(feats)
                _, pred = logits.max(dim=1)
                # _, pred = F.linear(feats, F.normalize(head_nodes_now, dim=-1)).max(1)
                # pred = dual_head_now.get_predicts(feats)
                # logits = head_nodes_now(feats)
                
                # _, pred = logits.max(1)
                targets = np.append(targets, y.cpu().numpy())
                preds_lb = np.append(preds_lb, pred.cpu().numpy())

            # preds_lb = preds.numpy()
            y = np.array(dl_whole_test_set.dataset.ys[dl_whole_test_set.dataset.uq_idxs])
            inc_new_selected_mask = np.zeros(len(y)).astype('bool')
            
            proj_all_new = utils.cluster_pred_2_gt(preds_lb.astype(int), y.astype(int))
            pacc_fun_all_new = partial(utils.pred_2_gt_proj_acc, proj_all_new)
            acc_a = pacc_fun_all_new(y.astype(int), preds_lb.astype(int))

            old_selected_mask = (y < nb_classes)    #* Base Session
            acc_o = pacc_fun_all_new(y[old_selected_mask].astype(int), preds_lb[old_selected_mask].astype(int))
            
            for new_cls in inc_new_classes:
                inc_new_selected_mask[np.where(y == new_cls)[0]] = True
            if sum(inc_new_selected_mask) > 0:
                acc_inc_new = pacc_fun_all_new(y[inc_new_selected_mask].astype(int), preds_lb[inc_new_selected_mask].astype(int))
            else:
                acc_inc_new = 0.
            # print(f"[Inference] Unknown Classes:{unknown_classes}/ {sum(inc_new_selected_mask)}")
            
            inc_old_selected_mask = ~(old_selected_mask + inc_new_selected_mask)    #* ~ (Base + Unknown): Only Incremental 
            if sum(inc_old_selected_mask) > 0:
                acc_inc_old = pacc_fun_all_new(y[inc_old_selected_mask].astype(int), preds_lb[inc_old_selected_mask].astype(int))
            else:
                acc_inc_old = 0.
        
        print('[Final Inference] Acc: All:{:.4f} / Base_classes:{:.4f} / Inc_old_classes:{:.4f} / Inc_new_classes:{:.4f}'.format(acc_a, acc_o, acc_inc_old, acc_inc_new))
        score_summary['All_ACC'] = acc_a
        score_summary['Old_ACC'] = acc_o
        score_summary['Inc_Old_ACC'] = acc_inc_old
        score_summary['Inc_New_ACC'] = acc_inc_new
        
    print("#"*100)
    print(f"[Seed {args.seed}] Performance Summary!!!")
    
    print("Old Discover ACC: {}".format(np.array(score_summary['Labeled_discover_li']).mean()))
    print("New Discover ACC: {}".format(np.array(score_summary['Unlabeled_discover_li']).mean()))
    print("All Discover ACC: {}".format(np.array(score_summary['All_discover_li']).mean()))
    print()
    print("Pseudo AVG Acc:", np.array(score_summary['pseudo_acc_avg']).mean())
    print("Pseudo Labeled Acc:", np.array(score_summary['pseudo_acc_labeled']).mean())
    print("Pseudo Unlabeled Acc:", np.array(score_summary['pseudo_acc_unlabeled']).mean())
    print()
    
    conf_lab = torch.cat(score_summary['confidence_labeled'], dim=0).detach().numpy()
    conf_unlab = torch.cat(score_summary['confidence_unlabeled'], dim=0).detach().numpy()
    
    print('pseudo confidence labeled [Avg/Min/Max]:', conf_lab.mean(), conf_lab.min(), conf_lab.max())
    print('pseudo confidence unlabeled [Avg/Min/Max]:', conf_unlab.mean(), conf_unlab.min(), conf_unlab.max())
    # 'confidence_labeled': [],
    # 'confidence_unlabeled': []
    
    
    print("-"*50)
    print('Base', score_summary['Base'])
    print('All_ACC', score_summary['All_ACC'])
    print('Old_ACC', score_summary['Old_ACC'])
    print('Inc_Old_ACC', score_summary['Inc_Old_ACC'])
    print('Inc_New_ACC', score_summary['Inc_New_ACC'])
    
    print("#"*100)

    #todo ================================================================================================
