import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import losses
import copy, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


class Soft_cross_entropy(object):
    def __call__(self, outputs_x, targets_x):
        return -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))


class SplitNet(torch.nn.Module):
    def __init__(self, sz_feature=512, sz_embed=128):
        torch.nn.Module.__init__(self)
        self.fc2 = nn.Linear(sz_feature, sz_embed // 2)
        self.batch2 = torch.nn.BatchNorm1d(sz_embed // 2)
        self.relu2 = nn.Sigmoid()
        self.fc3 = nn.Linear(sz_embed // 2, sz_embed // 2)
        self.batch3 = torch.nn.BatchNorm1d(sz_embed // 2)
        self.relu3 = nn.Sigmoid()
        self.fc6 = nn.Linear(sz_embed // 2, 2)

    def forward(self, X):
        out_f = self.fc2(X)
        out_f = self.batch2(out_f)
        out_f = self.relu2(out_f)
        out_f = self.fc3(out_f)
        out_f = self.batch3(out_f)
        out_f = self.relu3(out_f)
        out_f = self.fc6(out_f)

        return out_f


class SplitModlue:
    def __init__(self, save_path, sz_feature=512, sz_embed=64):
        self.siplitnet = SplitNet(sz_feature=sz_feature, sz_embed=sz_embed)
        self.siplitnet = self.siplitnet.cuda()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean').cuda()
        self.cross_entropy_none = nn.CrossEntropyLoss(reduction='none').cuda()
        self.soft_cross_entropy = Soft_cross_entropy()
        self.save_path=save_path

    def show_OnN(self, y, v, nb_classes, pth_result, pth_name, thres=0., is_hist=False, iter=0, loss=None):
        oo_i, on_i, no_i, nn_i = 0, 0, 0, 0
        o, n = [], []

        for j in range(len(y)):
            if y[j] < nb_classes:
                if loss is not None:
                    o.append(loss[j])
                else:
                    o.append(v[j])    
                
                if v[j] >= thres:
                    oo_i += 1
                else:
                    on_i += 1
            else:
                if loss is not None:
                    n.append(loss[j])
                else:
                    n.append(v[j])    

                if v[j] >= thres:
                    no_i += 1
                else:
                    nn_i += 1

        if is_hist is True:
            # plt.hist(o, bins=500, label='old', alpha=0.5)
            # plt.hist(n, bins=500, label='new', alpha=0.5)            
            plt.hist((o, n), histtype='bar', bins=100)
            # plt.savefig(self.save_path + '/' + 'Cos_similarity_' + str(step) + '.png')
            # plt.savefig(self.save_path + '/cross_entropy_GMM_' + str(epoch) + '_' + str(step) + '.png')
            plt.savefig(pth_result + '/' + pth_name + str(iter) + '.png')
            plt.close()
            # plt.clf()

            # for i in range(len(o)):
            #     Utils_SaveTxt(i, self.save_path, o[i], name='Cos_similarity_old')
            #     Utils_SaveTxt(i, self.save_path, o[i], name='cross_entropy_old'+ str(epoch) + '_' + str(step))
            # for i in range(len(n)):
            #     Utils_SaveTxt(i, self.save_path, n[i], name='Cos_similarity_new')
            #     Utils_SaveTxt(i, self.save_path, n[i], name='cross_entropy_new'+ str(epoch) + '_' + str(step))

        return oo_i, on_i, no_i, nn_i

    def predict_batchwise(self, model, dataloader):
        model_is_training = model.training
        model.eval()

        ds = dataloader.dataset
        A = [[] for i in range(len(ds[0]))]
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=True):
                for i, J in enumerate(batch):
                    if i == 0:
                        J = model(J.cuda())
                    for j in J:
                        A[i].append(j)
        model.train()
        model.train(model_is_training)

        return [torch.stack(A[i]) for i in range(len(A))]

    def predict_batchwise_split(self, model, splitnet, dataloader):
        model_is_training = model.training
        model.eval()
        splitnet.eval()

        ds = dataloader.dataset
        
        A = [[] for i in range(len(ds[0]))]
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=True):
                for i, J in enumerate(batch):
                    # print()
                    # print("[splitnet 150] J:", J.shape)
                    if i == 0:
                        J = model(J.cuda())
                        J = splitnet(J)
                    for j in J:
                        A[i].append(j)
        model.train()
        model.train(model_is_training)

        return [torch.stack(A[i]) for i in range(len(A))]

    def evaluate_cos_(self, model, dataloader):
        X, T, _ = self.predict_batchwise(model, dataloader)
        X = l2_norm(X)

        return X, T

    # generate_dataset(old_new_dataset_train, idx_rm, index_target=idx_o, target=y_p_noise, index_target_new=idx_n)
    def generate_dataset(self, dataset, index, index_target=None, target=None, index_target_new=None):
        dataset_ = copy.deepcopy(dataset)
        # print()
        # print("[splitNet-generate_dataset: 170] dataset_ sizs:", len(dataset_))
        # print("[splitNet-generate_dataset: 170] dataset_.ys:", len(dataset_.ys))
        # print("[splitNet-generate_dataset: 171] dataset_.uq_idxs:",dataset_.uq_idxs, len(dataset_.uq_idxs))
        # print()
        # print("[splitNet-generate_dataset: 170] index:",index, len(index))
        # print("[splitNet-generate_dataset: 170] index_target:",index_target, len(index_target))
        # print("[splitNet-generate_dataset: 170] index_target_new:", index_target_new, len(index_target_new))
        dataset_.binary = True
        dataset.uq_idxs = np.array(range(len(dataset_)))
        for i in range(len(dataset_.uq_idxs)):
            if i in index_target:
                dataset_.ys[i] = 0
            elif i in index_target_new:
                dataset_.ys[i] = 1
            else:
                dataset_.ys[i] = -1
        
        # print("[splitNet-generate_dataset: 197] dataset_ sizs:", len(dataset_))
        # print("[splitNet-generate_dataset: 198] dataset_.ys",len(dataset_.ys))
        # print("[splitNet-generate_dataset: 197] dataset_.uq_idxs:",dataset_.uq_idxs, len(dataset_.uq_idxs))
        
        

        return dataset_

    def calc_old_new(self, preds_cs, thres_cos, y, 
                     is_hist=True,
                     thres=0.5,
                     thres_min=0.1,
                     thres_max=0.5,
                     last_old_num=160,
                     step=0):

        if step > 0:
            print('Normalizing...')
            preds_cs = ((preds_cs - preds_cs.min()) / (preds_cs.max() - preds_cs.min()) )* 2.0 - 1.0

        thres_cos_min = thres_cos - 0.03
        thres_cos_max = thres_cos + 0.03
        # thres_cos_min = thres_cos - 0.001
        # thres_cos_max = thres_cos + 0.001

        idx_o_t = (preds_cs >= thres_cos_max)
        idx_o = np.nonzero(idx_o_t)[0]
        idx_n_t = (preds_cs < thres_cos_min)
        idx_n = np.nonzero(idx_n_t)[0]

        old_new = idx_o_t + idx_n_t
        old_new = ~old_new
        idx_rm = old_new.nonzero()[0]

        # oo_i, on_i, no_i, nn_i = 0, 0, 0, 0
        # for j in range(len(idx_o)):
        #     if y[idx_o[j]] < last_old_num:
        #         oo_i += 1
        #     else:
        #         no_i += 1
        # for j in range(len(idx_n)):
        #     if y[idx_n[j]] < last_old_num:
        #         on_i += 1
        #     else:
        #         nn_i += 1
        # print('Fine. Split result 1st. ({}~{})\t oo: {}\t on: {}\t no: {}\t nn: {}'.format(str(thres_cos_min), str(thres_cos_max), oo_i, on_i, no_i, nn_i))
        # print('idx_rm: ', len(idx_rm))

        # oo_i, on_i, no_i, nn_i = self.show_OnN(y, preds_cs, last_old_num, self.save_path, 'Cos_similarity_', thres=thres_cos, is_hist=is_hist, iter=0)
        # print('Fine. Split result 1st.(0.)\t oo: {}\t on: {}\t no: {}\t nn: {}'.format(oo_i, on_i, no_i, nn_i))

        return idx_o, idx_n, idx_rm


    def calc_GMM_cross(self, total_loss, y, 
                       is_hist=True,
                       thres=0.5,
                       thres_min=0.05,
                       thres_max=0.95,
                       last_old_num=160,
                       epoch=0,
                       training=True,
                       seed=1):

        total_loss_org = (total_loss - total_loss.min()) / (total_loss.max() - total_loss.min())
        total_loss = np.reshape(total_loss_org, (-1, 1))
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=seed)
        gmm.fit(total_loss)
        prob = gmm.predict_proba(total_loss)
        prob = prob[:, gmm.means_.argmin()]

        prob_mean = np.mean(prob)
        prob_var = np.var(prob)
        gmm_pro_max = thres_max
        gmm_pro_min = thres_min

        pred_zero = (prob >= gmm_pro_max)
        label_zero_index = pred_zero.nonzero()[0]  # old
        pred_one = (prob < gmm_pro_min)
        label_one_index = pred_one.nonzero()[0]  # new

        old_new = pred_zero + pred_one
        old_new = ~old_new
        idx_rm = old_new.nonzero()[0]

        oo_i, on_i, no_i, nn_i = self.show_OnN(y, prob, last_old_num, self.save_path, 'Fine_Split_', thres=thres, is_hist=is_hist, iter=epoch, loss=total_loss_org)
        print('Fine. Split result(0.5)\t oo: {}\t on: {}\t no: {}\t nn: {}'.format(oo_i, on_i, no_i, nn_i))

        discover_score={}
        # discover_score['old_detect_acc'] = oo_i / (oo_i+on_i)
        # discover_score['new_detect_acc'] = nn_i / (no_i+nn_i)
        if (oo_i+on_i) > 0:
            discover_score['old_detect_acc'] = oo_i / (oo_i+on_i)
        else:
            discover_score['old_detect_acc'] = 0.
        
        if (no_i+nn_i) > 0:
            discover_score['new_detect_acc'] = nn_i / (no_i+nn_i)
        else:
            discover_score['new_detect_acc'] = 0.
        
        discover_score['all_detect_acc'] = (oo_i+nn_i) / (oo_i+on_i+no_i+nn_i)
        
        
        if training:
            return label_zero_index, label_one_index, idx_rm, discover_score
        else:
            label_zero_index = np.concatenate([idx_rm, label_zero_index], axis=0)
            label_one_index = np.concatenate([idx_rm, label_one_index], axis=0)
            # return label_zero_index, label_one_index, idx_rm, discover_score
            return label_zero_index, label_one_index, prob

    def split_old_and_new(self,
                          main_model,
                          proxy,
                          old_new_dataset_eval,
                          old_new_dataset_train,
                          main_epoch=3,
                          sub_epoch=5,
                          lr=5e-5,
                          weight_decay=5e-3,
                          batch_size=64,
                          num_workers=4,
                          last_old_num=160,
                          thres_min=0.05,
                          thres_max=0.95,
                          thres_cos=0.0,
                          step=0,
                          seed=1):

        main_model_ = copy.deepcopy(main_model)
        main_model_ = main_model_.cuda()
        param_groups = [{'params': main_model_.parameters()}, {'params': self.siplitnet.parameters()}]
        opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        dl_ev = torch.utils.data.DataLoader(old_new_dataset_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dl_uq_idxs = old_new_dataset_eval.uq_idxs
        
        m_org, y_true = self.evaluate_cos_(main_model, dl_ev)
        cos_sim = F.linear(losses.l2_norm(m_org), losses.l2_norm(proxy))
        v, y_p = torch.max(cos_sim, dim=1)

        y_p_noise = y_p.cpu().detach().numpy()
        v_arr = v.cpu().detach().numpy()
        y_true_arr = y_true.cpu().detach().numpy()

        idx_o, idx_n, idx_rm = self.calc_old_new(v_arr, thres_cos, y_true_arr, last_old_num=last_old_num, thres_min=thres_min, thres_max=thres_max, step=step)

        # print(f"[Splitnet L321:] old_new_dataset_train.uq_idxs: ({len(old_new_dataset_train.uq_idxs)})", old_new_dataset_train.uq_idxs)
        ev_dataset_o = self.generate_dataset(old_new_dataset_train, idx_rm, index_target=idx_o, target=y_p_noise, index_target_new=idx_n)
        # print("[Splitnet L321:] ev_dataset_o:", len(ev_dataset_o))
        # print(f"[Splitnet L321:] ev_dataset_o.uq_idxs: ({len(ev_dataset_o.uq_idxs)})", ev_dataset_o.uq_idxs)
        ev_dataset_o.binary= True
        dl_tr_o = torch.utils.data.DataLoader(ev_dataset_o, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

        for epoch_main in range(main_epoch):
            for epoch_sub in range(sub_epoch):
                pbar = enumerate(dl_tr_o)
                total_loss_arr = []
                #* for batch_idx, (x, y,_, z) in pbar:
                for batch_idx, (x, y, z) in pbar:
                    # print("[splitNet: 320] z:", z)
                    self.siplitnet.train()
                    main_model_.train()
                    y_mask = y>=0
                    y = y[y_mask]
                    x = x[y_mask]
                    
                    y = F.one_hot(y, num_classes=2).cuda()
                    x = x.cuda()
                    # print('[splitnet L:326] x:', x.shape)
                    m = main_model_(x)
                    # print('[splitnet L:326] m:', m.shape)
        
                    y_bin_pridict = self.siplitnet(m)
                    total_loss = self.soft_cross_entropy(y_bin_pridict, y)

                    opt.zero_grad()
                    total_loss.backward()
                    opt.step()

                    total_loss_arr.append(total_loss.cpu().detach().numpy())
                total_loss_arr = np.array(total_loss_arr)
                if len(total_loss_arr)>0:
                    avg_loss = np.mean(total_loss_arr)
                else:
                    avg_loss = 0.
                print('Fine. Split result Epoch(main/sub): {}/{} Loss: {}'.format(epoch_main, epoch_sub, avg_loss))

            m_, y_true, _ = self.predict_batchwise_split(main_model_, self.siplitnet, dl_ev)

            y_true_arr = y_true.cpu().detach().numpy()
            m_ = torch.softmax(m_, dim=1)
            y_0_1 = []
            for i in range(len(y_p_noise)):
                y_0_1.append(0)

            y_hot = np.array(y_0_1)
            y_hot = torch.tensor(y_hot)
            y_hot = y_hot.cuda()
            y_hot = y_hot.long()

            loss_total = self.cross_entropy_none(m_, y_hot)

            idx_o, idx_n, idx_rm, discover_score = self.calc_GMM_cross(loss_total.cpu().detach().numpy(), y_true_arr, epoch=epoch_main, last_old_num=last_old_num, thres_min=thres_min, thres_max=thres_max, seed=seed)
            ev_dataset_o = self.generate_dataset(old_new_dataset_train, idx_rm, index_target=idx_o, target=y_p_noise, index_target_new=idx_n)
            dl_tr_o = torch.utils.data.DataLoader(ev_dataset_o, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

        idx_o, idx_n, _, discover_score = self.calc_GMM_cross(loss_total.cpu().detach().numpy(), y_true_arr, epoch=epoch_main, last_old_num=last_old_num, thres_min=thres_min, thres_max=thres_max, training=True, seed=seed)
        del opt
        
        return idx_o, idx_n, discover_score