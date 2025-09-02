# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 data,
                 args,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device
        self._data = data
        self._args = args
        self.max_noise = args['max_noise']
        self.state_dim = data.xdim# + data.ydim
        self.action_dim = data.ydim# - data.nknowns

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        input_dim = self.state_dim + self.action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_layer = nn.Linear(256, self.action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x).clamp(min=-self.max_noise, max=self.max_noise)


class MLP4FM(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 data,
                 args,
                 device,
                 t_dim=16):

        super(MLP4FM, self).__init__()
        self.device = device
        self._data = data
        self._args = args
        self.state_dim = data.xdim# + data.ydim
        self.action_dim = data.ydim# - data.nknowns

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        input_dim = self.state_dim + self.action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_layer = nn.Linear(256, self.action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)