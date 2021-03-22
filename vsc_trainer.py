"""
    VSC Trainer

    Created on 2020.08.03.
    @author: Younghyun Kim
"""
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from tensorboardX import SummaryWriter

from latent_regime.utils import save_model, save_vars

class VSCTrainer:
    """
        VSC Trainer Class
    """
    def __init__(self, model, load_model=False,
                 filepath="./models/", logdir="./logdir/",
                 modelname=None,
                 train_dataset=None,
                 test_dataset=None, clip_grad=0.5,
                 K=1, recon_K=0, lr=0.001, amsgrad=True,
                 beta1=0.9, beta2=0.999, device='cuda:0',
                 kl_coeff=1., spars_coeff=1.):
        """ Initialization """
        self.model = model  # VSC
        self.load_model = load_model
        self.filepath = filepath
        self.logdir = logdir
        self.device = device

        self.kl_coeff = kl_coeff
        self.spars_coeff = spars_coeff

        self.trainset = None
        self.testset = None

        self.get_datasets(train_dataset)

        self.K = K
        self.recon_K = recon_K

        self.lr = lr
        self.amsgrad = amsgrad
        self.beta1 = beta1
        self.beta2 = beta2
        self.clip_grad = clip_grad

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.model.parameters()),
                                          lr=lr, amsgrad=amsgrad,
                                          betas=(beta1, beta2))

        self.model = self.model.to(self.device)

        if load_model:
            self.load_our_model(filepath + modelname)

        self.train_results = defaultdict(list)

        self.model_file_name = None

    def get_datasets(self, trainset=None):
        " get datasets "

        if trainset is not None:
            trainset = torch.FloatTensor(trainset.astype(float))
        self.trainset = trainset

    def load_our_model(self, file_name):
        " load model "
        self.model.load_state_dict(torch.load(file_name,
                                              map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def save_our_model(self, file_name=None):
        " save model "
        if file_name is not None:
            save_model(self.model, file_name)
        else:
            today = datetime.datetime.today().strftime('%Y%m%d')
            file_name = self.filepath + self.model.modelName + \
                    "_" + today + ".pt"
            save_model(self.model, file_name)
        self.model_file_name = file_name

    def calculate_cost(self, x, alpha, lmbda, nu, K=1, pos=False, eps=1e-6):
        """
            Calculation for cost
        """
        alpha = torch.tensor(alpha).to(self.device)
        z_mu, z_sig, log_eta = self.model.get_z_params(x)
        z_scores, _ = self.model._calc_z_from_params(z_mu, z_sig, log_eta,
                                                     lmbda=lmbda, K=K,
                                                     num=nu)
        z_mu_p, z_sig_p, log_eta_p =\
                self.model.get_z_params(self.model.pseudo_inputs)

        sel = self.model._calc_selection(x)

        # Reconstruction Loss
        px_z = self.model.px_z(*self.model.dec(z_scores, pos))
        lpx_z = px_z.log_prob(x).sum(-1)

        # latent loss
        z_size = z_mu.shape

        sel = sel.unsqueeze(2)
        sel = sel.repeat(1, 1, z_size[1])
        sel = sel.transpose(2, 1)

        zm = z_mu.unsqueeze(2)
        zm = zm.repeat(1, 1, self.model.n_pinputs)
        zlss = z_sig.unsqueeze(2)
        zlss = zlss.repeat(1, 1, self.model.n_pinputs)
        zle = log_eta.unsqueeze(2)
        zle = zle.repeat(1, 1, self.model.n_pinputs)
        zpm = z_mu_p.unsqueeze(2)
        zpm = zpm.repeat(1, 1, z_size[0])
        zpm = zpm.transpose(2, 0)
        zplss = z_sig_p.unsqueeze(2)
        zplss = zplss.repeat(1, 1, z_size[0])
        zplss = zplss.transpose(2, 0)
        zple = log_eta_p.unsqueeze(2)
        zple = zple.repeat(1, 1, z_size[0])
        zple = zple.transpose(2, 0)

        # KL(q(z|x)||q(z|x_p)) KL(1||2)
        v_mean = zpm  #2
        aux_mean = zm  #1

        v_log_sig_sq = torch.log(torch.exp(zplss) + eps)  #2
        aux_log_sig_sq = torch.log(torch.exp(zlss) + eps)  #1
        v_log_sig = torch.log(torch.sqrt(torch.exp(v_log_sig_sq) + eps) + eps)  #2
        aux_log_sig = torch.log(torch.sqrt(torch.exp(aux_log_sig_sq) + eps) + eps)  #1
        cost_KLN_a = v_log_sig - aux_log_sig +\
                (torch.exp(aux_log_sig_sq) +\
                 ((aux_mean - v_mean) ** 2)) /\
                (2 * torch.exp(v_log_sig_sq)) - 0.5
        cost_KLN_b_1 = (torch.exp(zle) * cost_KLN_a) + sel
        cost_KLN_b_2 = cost_KLN_b_1.mean(2)
        cost_KLN_b = cost_KLN_b_2 / (sel.mean(2) + eps)

        zpe = torch.exp(zple)
        cost_KLS_a = (torch.log(1. - zpe + eps) -\
                        torch.log(1. - torch.exp(zle) + eps)) *\
                        (1 - torch.exp(zle)) +\
                        ((zple - zle) * torch.exp(zle))
        cost_KLS_b = -((cost_KLS_a * sel).mean(2) / (sel.mean(2) + eps))

        cost_KL = (cost_KLN_b + cost_KLS_b).sum(1)

        # KL Average qp
        log_eta_p_mean_1 = torch.log(((torch.exp(zple) * sel).mean(2) + eps) /\
                (sel.mean(2) + eps))
        log_eta_p_mean = torch.log(torch.exp(log_eta_p_mean_1).mean(1) + eps)

        latent_loss_spike2_1 = torch.log(1. - alpha) -\
                torch.log(1 - torch.exp(log_eta_p_mean) + eps)
        latent_loss_spike2_2 = latent_loss_spike2_1 * \
                (1. - torch.exp(log_eta_p_mean))
        latent_loss_spike2_3 = latent_loss_spike2_2 + \
                ((torch.log(alpha) - log_eta_p_mean) * \
                                torch.exp(log_eta_p_mean))
        latent_loss_spike2 = -self.model.z_dim * latent_loss_spike2_3

        cost_KLP = latent_loss_spike2.mean()

        cost = (-lpx_z + self.kl_coeff * cost_KL).mean() +\
                self.spars_coeff * cost_KLP
        recon_loss = lpx_z.mean().detach()
        cost_div = cost_KL.mean().detach()
        sd = cost_KLP.detach()

        return cost, sd, recon_loss, cost_div

    def train_random(self, alpha=0.9, nu=100,
                     epochs=20, K=1, pos=True, batch_size=64,
                     wms=0, wme=20000, sche=100000,
                     val_prob=0.2, val_batch=32,
                     seed=0):
        """
            train random method
            * 시점에 상관없이 무작위 학습, 주식 임베딩에 적합,
            * latent regime을 위해서는 시점을 고려한 train 권장

            wms: warm-up-start
            wme: warm-up-end
        """
        writer = SummaryWriter(self.logdir)
        torch.manual_seed(seed)
        np.random.seed(seed)

        scheduler = optim.lr_scheduler.StepLR(self.optimizer, sche, gamma=0.9)

        # Set random train index
        train_idx = np.arange(self.trainset.shape[0])
        np.random.shuffle(train_idx)

        train_N = train_idx.shape[0]
        val_N = round(train_N * val_prob)

        val_idx = np.random.choice(train_idx, val_N, replace=False)

        train_idx = np.setdiff1d(train_idx, val_idx)
        train_N = train_idx.shape[0]

        length = train_N // batch_size
        val_length = val_N // val_batch

        lam = 0
        lamf = 1
        lam_cnt = 0
        wml = wme - wms

        for epoc in range(epochs):
            self.model.train()

            obj_loss = 0.
            recon_loss = 0.
            sd_loss = 0.
            kl_loss = 0.

            for j in range(length):
                train_i = train_idx[j * batch_size:(j + 1) * batch_size]
                trainset = self.trainset[train_i]
                trainset = trainset.to(self.device)

                self.optimizer.zero_grad()

                if lam_cnt < wme and lam_cnt >= wms:
                    lam = lam + lamf / wml
                    lam_cnt += 1

                cost, sd, rec_loss, div_loss =\
                        self.calculate_cost(trainset, alpha, lam, nu,
                                            K=K, pos=pos)
                cost.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                scheduler.step()
                obj_loss += -cost.detach().item()
                recon_loss += rec_loss.detach().item()
                kl_loss += div_loss.detach().item()
                sd_loss += sd.detach().item()
                print(j, end='\r')
            print("\n", epoc)

            writer.add_scalar('train_obj_loss', obj_loss / train_N * batch_size, epoc)
            writer.add_scalar('train_recon_loss', recon_loss / train_N * batch_size, epoc)
            writer.add_scalar('train_kl_loss', kl_loss / train_N * batch_size, epoc)
            writer.add_scalar('train_sd_loss', sd_loss / train_N * batch_size, epoc)

            # validation
            if val_N > 0:
                self.model.eval()

                vobj_loss = 0.
                vrecon_loss = 0.
                vsd_loss = 0.
                vkl_loss = 0.

                for j in range(val_length):
                    vtrain_i = val_idx[j * val_batch:(j + 1) * val_batch]
                    valset = self.trainset[vtrain_i]
                    valset = valset.to(self.device)
                    with torch.no_grad():
                        val_cost, val_sd, val_rec_loss, val_div_loss =\
                                self.calculate_cost(valset, alpha, lam, nu,
                                                    K=K, pos=pos)
                        vobj_loss += -val_cost.item()
                        vrecon_loss += val_rec_loss.item()
                        vsd_loss += val_sd.item()
                        vkl_loss += val_div_loss.item()

                writer.add_scalar('val_obj_loss', vobj_loss / val_N * val_batch, epoc)
                writer.add_scalar('val_recon_loss', vrecon_loss / val_N * val_batch, epoc)
                writer.add_scalar('val_kl_loss', vkl_loss / val_N * val_batch, epoc)
                writer.add_scalar('val_sd_loss', vsd_loss / val_N * val_batch, epoc)

            self.save_our_model()
        save_vars(self.train_results,
                  self.model_file_name[:-3] + "_losses.rar")
