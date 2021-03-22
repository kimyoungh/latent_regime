"""
    국면 분석 모델을 위한 Variational Sparse Coding Model(By PyTorch)

    Created on 2020.07.17.
    @author: Younghyun Kim
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform

import pdb

class VSC(nn.Module):
    """ VSC Class """
    def __init__(self, input_dim=120, n_pinputs=16,
                 hidden_dim=64, z_dim=32,
                 likelihood_dist=Normal,
                 model_name='VSC'):
        " Initialization "
        super(VSC, self).__init__()

        self.input_dim = input_dim
        self.n_pinputs = n_pinputs
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.modelName = model_name

        self.px_z = likelihood_dist
        self.slab = Normal
        self.spike = Uniform

        self.enc = Encoder(input_dim, hidden_dim, z_dim)
        self.dec = Decoder(input_dim, hidden_dim, z_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.selection_layer = nn.Linear(input_dim, n_pinputs)
        self.pinputs_net = nn.Embedding(n_pinputs, input_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    @property
    def pseudo_inputs(self):
        " Get pseudo inputs data "
        return self.pinputs_net.weight

    def _calc_reconstruct(self, z, K=0, pos=False):
        " reconstruction "

        if K == 0:
            recon = self.dec(z, pos)[0]
        else:
            px_z = self.px_z(*self.dec(z, pos))
            recon = px_z.rsample(torch.Size([K]))
            if K > 1:
                recon = recon.mean(0)
            else:
                recon = recon.squeeze(0)

        if pos:
            recon = self.relu(recon)

        return recon

    def get_z(self, x, K=1, num=100, dis=True):
        with torch.no_grad():
            z_scores, log_eta = self._get_z(x, K, num, dis=dis)
            z_scores = z_scores.detach()
            log_eta = log_eta.detach()

        return z_scores, log_eta

    def _get_z(self, x, K=1, num=100, dis=True):
        " Get z by sampling from posterior "
        if dis:
            if K == 0:  # For evaluation, not for training
                z_mu, _, log_eta = self.enc(x)

                msk = torch.round(torch.exp(log_eta))

                z_scores = z_mu * msk

            else:
                mu, sig, log_eta = self.enc(x)
                slab = self.slab(mu, sig)
                z_slab = slab.rsample(torch.Size([K]))

                if K > 1:
                    z_slab = z_slab.mean(0)
                else:
                    z_slab = z_slab.squeeze(0)

                msk = torch.round(torch.exp(log_eta))

                z_scores = z_slab * msk
        else:
            if K == 0:  # For evaluation, not for training
                z_mu, _, log_eta = self.enc(x)

                spike = self.spike(0., 1.)
                z_spike = spike.rsample(z_mu.shape)
                z_spike = z_spike.to(self.device)

                msk = self.sigmoid(num * (z_spike.detach() \
                        - (1. - torch.exp(log_eta))))

                z_scores = z_mu * msk

            else:
                mu, sig, log_eta = self.enc(x)
                slab = self.slab(mu, sig)
                spike = self.spike(0., 1.)

                z_slab = slab.rsample(torch.Size([K]))
                z_spike = spike.rsample(z_slab.shape)
                z_spike = z_spike.to(self.device)

                log_eta_ = log_eta.unsqueeze(0).repeat(K, 1, 1)

                msk = self.sigmoid(num * (z_spike.detach() \
                        - (1. - torch.exp(log_eta_))))

                z_scores = z_slab * msk

                if K > 1:
                    z_scores = z_scores.mean(0)
                else:
                    z_scores = z_scores.squeeze(0)

        return z_scores, log_eta

    def _calc_z_from_params(self, z_mu, z_sig, log_eta,
                            lmbda=1., K=1, num=100):
        " Get z from z params "
        mu_zeros = torch.zeros_like(z_mu).to(self.device)
        sig_zeros = torch.zeros_like(z_sig).to(self.device)

        mu = lmbda * z_mu + (1. - lmbda) * mu_zeros
        sig = lmbda * z_sig + (1. - lmbda) * sig_zeros

        slab = self.slab(mu, sig)

        spike = self.spike(0., 1.)

        z_slab = slab.rsample(torch.Size([K]))
        z_spike = spike.rsample(z_slab.shape)
        z_spike = z_spike.to(self.device)

        log_eta_ = log_eta.unsqueeze(0).repeat(K, 1, 1)

        msk = self.sigmoid(num * (z_spike.detach() \
                - (1 - torch.exp(log_eta_))))

        z_scores = z_slab * msk

        if K > 1:
            z_scores = z_scores.mean(0)
        else:
            z_scores = z_scores.squeeze(0)

        return z_scores, log_eta

    def get_z_params(self, x):
        " get z params "
        mu, sig, log_eta = self.enc(x)

        return mu, sig, log_eta

    def _calc_selection(self, x, eps=1e-6, a=60., b=0.5):
        """
            selection function
        """
        selection_ = self.selection_layer(x)

        v_norm = torch.norm(selection_, dim=-1).detach()
        s_norm = v_norm.repeat((self.n_pinputs, 1)).transpose(0, 1)

        selection = selection_ / (s_norm + eps)
        sel = self.sigmoid(a * (selection - b))

        return sel

    @property
    def device(self):
        return self.pseudo_inputs.device

    def forward(self, x, K=1, num=100, dis=True,
                recon_K=0, pos=True):
        z_scores, log_eta = self.get_z(x, K=K, num=num, dis=dis)
        recon = self._calc_reconstruct(z_scores, K=recon_K, pos=pos)

        return z_scores, recon.detach(), log_eta

class Encoder(nn.Module):
    " Encoder class "
    def __init__(self, input_dim=120, hidden_dim=64,
                 z_dim=32):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_sig = nn.Linear(hidden_dim, z_dim)
        self.fc_gamma = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.gelu(hidden)

        z_mu = self.fc_mu(hidden)
        z_sigma = self.softplus(self.fc_sig(hidden))
        log_gamma = -self.relu(-self.fc_gamma(hidden))

        return z_mu, z_sigma, log_gamma


class Decoder(nn.Module):
    " Decoder class "
    def __init__(self, output_dim=120, hidden_dim=64,
                 z_dim=32):
        super(Decoder, self).__init__()

        self.output_dim = output_dim 
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sig = nn.Linear(hidden_dim, output_dim)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z, pos=False):
        hidden = self.fc1(z)
        hidden = self.gelu(hidden)

        if pos:
            x_mu = self.relu(self.fc_mu(hidden))
        else:
            x_mu = self.fc_mu(hidden)
        x_sigma = self.softplus(self.fc_sig(hidden))

        return x_mu, x_sigma
