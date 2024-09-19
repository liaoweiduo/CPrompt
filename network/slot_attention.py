import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class SlotAttention(nn.Module):
    def __init__(self, emb_d, n_slots, key_dim=128, n_iter=5):
        super().__init__()
        self.emb_d = emb_d          # emb for representation 768
        self.key_d = key_dim        # emb for slot: Dslot 64

        # slot basic param
        self.n_slots = n_slots  # 5   number of slots
        self.n_iter = n_iter     # T
        self.attn_epsilon = 1e-8
        self.gru_d = self.key_d

        # slot related modules
        self.ln_input = nn.LayerNorm(self.emb_d)
        self.ln_slot = nn.LayerNorm(self.key_d)
        self.ln_output = nn.LayerNorm(self.key_d)
        self.mu = init_tensor(1, 1, self.key_d)             # slot gaussian distribution mu
        self.log_sigma = init_tensor(1, 1, self.key_d)          # slot gaussian distribution sigma
        self.k = nn.Linear(emb_d, key_dim, bias=False)
        self.q = nn.Linear(key_dim, key_dim, bias=False)
        self.v = nn.Linear(emb_d, key_dim, bias=False)
        # self.gru = nn.GRU(self.key_d, self.gru_d, batch_first=True)
        self.gru = nn.GRUCell(self.key_d, self.gru_d)
        self.mlp = nn.Sequential(
            nn.Linear(self.key_d, self.key_d, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.key_d, self.key_d, bias=True)
        )

        # slot decoder
        self.ln_decoder = nn.LayerNorm(self.key_d)
        self.decoder = nn.Sequential(
            nn.Linear(self.key_d, self.key_d * 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.key_d * 2, self.emb_d, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_d, self.emb_d, bias=True),
        )

    def forward(self, features, all_loss=False):
        # features: [bs, n196, 768]
        slots, attn = self.forward_slots(features)
        # slots [bs, k20, d64], attn [bs, n196, k20]

        # recon
        slot_features = self.ln_decoder(slots)
        slot_features = self.decoder(slot_features)     # [bs, k20, 768]
        slot_features = torch.einsum('bkd,bnk->bnd', slot_features, attn)       # [bs, n196, 768]

        # recon loss
        recon_loss = F.mse_loss(slot_features, features, reduction='none')
        if all_loss:
            recon_loss = recon_loss.mean(dim=-1).mean(dim=-1)       # [bs]
        else:
            recon_loss = recon_loss.mean()
        return slots, attn, recon_loss

    def forward_slots(self, features):
        # features [bs, 196, 768]
        bs = features.shape[0]

        # init
        features = self.ln_input(features)
        slots = torch.randn(bs, self.n_slots, self.key_d, device=self.log_sigma.device) * torch.exp(self.log_sigma) + self.mu
        # [bs, k, 64]
        attn_vis = None

        # iter
        k = self.k(features)    # [bs, 196, 64]
        v = self.v(features)    # [bs, 196, 64]
        k = (self.key_d ** (-0.5)) * k

        for t in range(self.n_iter):
            slots_prev = slots.clone()
            slots = self.ln_slot(slots)
            q = self.q(slots)       # [bs, k, 64]

            # b = bs, n = 196, k = 5, d = 64
            ## softmax(KQ^T/sqrt(d), dim='slots')
            # sum((b x n x 1 x d) * [b x 1 x k x d]) = (b x n x k)
            attn = torch.einsum('bnd,bkd->bnk', k, q)
            # attn = attn * (self.key_d ** -0.5)
            # softmax over slots
            attn_vis = F.softmax(attn, dim=-1)      # [b, n, k]

            ## updates = WeightedMean(attn+epsilon, v)
            attn = attn_vis + self.attn_epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            # sum((b x n x k x 1) * (b x n x 1 x d)) = (b x k x d)
            updates = torch.einsum('bnk,bnd->bkd', attn, v)

            ## slots = GRU(state=slots_prev[b,k,d], inputs=updates[b,k,d])  (for each slot)

            slots = self.gru(updates.view(-1, self.key_d),               # [b*k, d]
                             slots_prev.reshape(-1, self.key_d))         # [b*k, d]
            # slots = self.gru(updates.view(-1, 1, self.key_d).contiguous(),       # [b*k, 1, d]
            #                  slots_prev.view(1, -1, self.key_d).contiguous()            # [1, b*k, d]
            #                  )[0]        # out: [b*k, 1, d]
            slots = slots.view(bs, self.n_slots, self.key_d)        # [b, k, d]

            ## slots += MLP(LayerNorm(slots))
            slots = slots + self.mlp(self.ln_output(slots))

        return slots, attn_vis

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            param.grad = None

    def _load_model(self, filename, freeze=False):
        logging.info(f'=> Load slot from {filename}')
        state_dict = torch.load(filename + 'class.pth')
        # complete with/without module.
        for key in list(state_dict.keys()):
            if 'slot_attn' in key:
                t = state_dict.pop(key, None)
                state_dict[key[10:]] = t      # remove slot_attn.
            else:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=True)
        logging.info(f'=> Load Done with params: {list(state_dict.keys())}.')

        self.eval()

        if freeze:
            self.freeze()


def init_tensor(a, b=None, c=None, d=None, ortho=False):
    if b is None:
        p = torch.nn.Parameter(torch.FloatTensor(a), requires_grad=True)
    elif c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    elif d is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c, d), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.xavier_uniform_(p)
    return p


class Slot2Prompt(nn.Module):
    def __init__(self, emb_d, key_dim=128, selector_mode='gate', mode_params=None):
        super().__init__()
        if mode_params is None:
            mode_params = dict(e_pool_size=100, e_p_length=8)
        self.emb_d = emb_d
        self.key_d = key_dim

        # prompt basic param
        self.e_p_length = mode_params['e_p_length']    # 8
        self.prompt_map = nn.Sequential(
            nn.Linear(self.key_d, 2*self.key_d), nn.ReLU(inplace=True),
            nn.Linear(2*self.key_d, self.e_p_length * self.emb_d))

        self.selector_mode = selector_mode     # [gate, mlp, attn]
        if self.selector_mode == 'gate':
            self.slot_map = nn.Linear(self.key_d, 1)
        elif self.selector_mode == 'mlp':
            self.slot_map = nn.Linear(self.key_d, self.key_d)
        else:
            raise NotImplementedError

    def forward(self, slots, s2p=None, train=False):
        # slots [bs, n20, h64]
        bs, n, h = slots.shape
        if s2p is None:
            s2p = self

        if self.selector_mode == 'gate':
            slot_map = s2p.slot_map          # [self.key_d -> self.key_d] or -> 1
            prompt_map = s2p.prompt_map      # [self.key_d -> self.e_p_length * self.emb_d]
            weights = F.sigmoid(slot_map(slots))        # -> [bs, k, 1]
            weighted_slots = torch.sum(weights * slots, dim=1)     # -> [bs, h]
            prompts = prompt_map(weighted_slots).reshape(bs, self.e_p_length, self.emb_d)
            # [bs, l, d]
        elif self.selector_mode == 'mlp':       # use dense
            slot_map = s2p.slot_map          # [self.key_d -> self.key_d] or -> 1
            prompt_map = s2p.prompt_map      # [self.key_d -> self.e_p_length * self.emb_d]
            weighted_slots = slot_map(slots)
            weighted_slots = torch.mean(weighted_slots, dim=1)   # mean over K
            prompts = prompt_map(weighted_slots).reshape(bs, self.e_p_length, self.emb_d)
            # [bs, l, d]
        else:
            raise NotImplementedError

        # prompt_map = getattr(self, f's2p')  # [h64, e12, p8, d768]
        # # [bs, k20, h64] @ [h64, e12, p8, d768] -> [bs, k20, e12, p8, d768]
        # prompts = torch.einsum('bkh,hepd->bkepd', slots, prompt_map)

        return prompts

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):  # 施密特正交化

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        if self.FPS:        # use all prompts
            s = 0
            f = self.e_pool_size
        else:
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    # def prompt_map_init(self, task_id):
    #     for e in self.e_layers:
    #         prompt_map = init_tensor(self.key_d, self.e_p_length, self.emb_d)  # [64, 8, 768]
    #         # setattr(self, f's2p_{task_id}_{e}', prompt_map)       # [bs, 64] @ [64, 8, 768] -> [bs, 8, 768]
    #         setattr(self, f's2p_{task_id}_{e}', prompt_map)

    # def prompt_map_freeze(self, task_id):
    #     for e in self.e_layers:
    #         prompt_map = getattr(self, f's2p_{task_id}_{e}')
    #         prompt_map.requires_grad = False
    #         setattr(self, f's2p_{task_id}_{e}', prompt_map)
