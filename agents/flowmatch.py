# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


from agents.helpers import Losses
# from utils.utils import Progress, Silent


class FlowMatching(nn.Module):
    def __init__(self, data, model, args):
        super(FlowMatching, self).__init__()

        self.state_dim = data.xdim + data.ydim
        self.action_dim = data.ydim
        self.action_partial_dim = data.ydim - data.neq
        self.model = model
        self.data = data

        # Flow Matching specific parameters
        self.n_timesteps = int(args.get('n_timesteps', 5))
        # Flow Matching uses continuous time t in [0, 1]
        self.time_scale = args.get('time_scale', 1.0)
        # Time scheduling
        self.default_schedule_type = args.get('schedule_type', 'uniform')
        # Annealing ratio for annealing schedule (default: 4, meaning 1/4)
        self.annealing_ratio = int(args.get('annealing_ratio', 4.0))
        # Loss function for Flow Matching
        self.loss_fn = Losses[args.get('loss_type', 'l2')]()

    # ------------------------------------------ sampling ------------------------------------------#

    def get_flow_path(self, x_start, x_end, t):
        """
        Get the flow path from x_start to x_end at time t. Linear interpolation: x_t = (1-t) * x_start + t * x_end
        """
        # Ensure t has the right shape for broadcasting: (batch_size,) -> (batch_size, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return (1 - t) * x_start + t * x_end

    def get_velocity_field(self, x_start, x_end, t):
        """
        Get the true velocity field (derivative of the flow path). For linear interpolation: v_t = x_end - x_start
        """
        return x_end - x_start

    def get_time_schedule(self, batch_size, device, schedule_type='uniform'):
        """
        Generate time schedule for flow matching
        Args:
            batch_size: number of samples
            device: torch device
            schedule_type: 'uniform', 'random', or 'annealing'
        Returns:
            t_schedule: tensor of shape (batch_size, n_timesteps) with time values
        """
        if schedule_type == 'uniform':
            # Uniform linear schedule: t = [0, 1/n, 2/n, ..., (n-1)/n]
            # Expand to (batch_size, n_timesteps) - same schedule for all data points
            t_schedule = torch.linspace(0, 1, self.n_timesteps+1, device=device)
            t_schedule = t_schedule.unsqueeze(0).expand(batch_size, -1)
            
        elif schedule_type == 'random':
            # Random schedule: sample t values randomly from [0, 1] for each data point
            # Start at 0, then random values (not necessarily ending at 1)
            random_vals = torch.rand(batch_size, self.n_timesteps - 1, device=device)
            random_vals = torch.sort(random_vals, dim=1)[0]
            # Concatenate: [0, sorted_random_values]
            t_schedule = torch.cat([
                torch.zeros(batch_size, 1, device=device),  # Start at 0
                random_vals,
                torch.ones(batch_size, 1, device=device)  # End at 1
            ], dim=1)
            
        elif schedule_type == 'annealing':
            # Annealing schedule: steps ∝ 1 / (1 + k * n), k = 0..(total_steps-1)
            # Output n_timesteps+1 time points from 0 to 1 (inclusive), with strictly decreasing steps.
            n_param = float(getattr(self, 'annealing_ratio', 4.0))
            n_param = float(max(n_param, 1e-6))

            total_steps = self.n_timesteps
            ks = torch.arange(total_steps, device=device, dtype=torch.float32)
            weights = 1.0 / ((1.0 + ks) * n_param)
            steps = weights / weights.sum()

            t_values = torch.cat([
                torch.tensor([0.0], device=device, dtype=steps.dtype),
                torch.cumsum(steps, dim=0)
            ], dim=0)
            t_values[-1] = 1.0  # numerical safety

            t_schedule = t_values.unsqueeze(0).expand(batch_size, -1)
            
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}. Use 'uniform', 'random', or 'annealing'")
        return t_schedule

    def p_sample_loop(self, state, x_init, return_flow=False):
        """
        Sampling loop using Flow Matching
        Always starts from x_init (known start point) and flows to target
        """
        device = state.device
        batch_size = state.shape[0]
        
        # Always use the provided initial point
        x = x_init.to(device)
        if return_flow:
            flow_path = [x]

        # Generate time schedule
        t_schedule = self.get_time_schedule(batch_size, device, self.default_schedule_type)
        # Now t_schedule is always (batch_size, n_timesteps)
        # Integrate the velocity field from t=0 to t=1
        for i in range(self.n_timesteps):
            # Get time for each data point: t_schedule[batch_idx, step_idx]
            t = t_schedule[:, i]  # Shape: (batch_size,)
            
            # Predict velocity field
            velocity_pred = self.model(x, t, state)
            
            # Update x using predicted velocity (Euler integration)
            dt = t_schedule[:, i+1] - t_schedule[:, i]  # Shape: (batch_size,)
            # Expand dt for broadcasting: (batch_size,) -> (batch_size, action_dim)
            dt_expanded = dt.unsqueeze(1).expand(-1, x.shape[1])
            x_next = x + dt_expanded * velocity_pred
            
            x = x_next
            
            if return_flow:
                flow_path.append(x)
        
        if return_flow:
            return x, torch.stack(flow_path, dim=1)
        else:
            return x

    def sample(self, state, action_init=None, *args, **kwargs):
        """
        Main sampling function. Always requires action_init (start point)
        """
        if action_init is None:
            raise ValueError("FlowMatching.sample() requires action_init parameter (start point)")
        action = self.p_sample_loop(state, action_init, *args, **kwargs)
        # Complete partial actions if needed
        action = self.data.complete_partial_parallel(state, action[:, :self.action_partial_dim])
        return action

    # ------------------------------------------ training ------------------------------------------#

    def get_training_sample(self, x_start, x_end, t):
        """
        Get training sample at time t along the flow path
        x_start: known start point, x_end: known target point
        """
        x_t = self.get_flow_path(x_start, x_end, t)
        v_t = self.get_velocity_field(x_start, x_end, t)
        return x_t, v_t

    def loss(self, x_start, x_target, state, weights=1.0):
        """
        Flow Matching loss with explicit start and target points
        x_start: known start point
        x_target: known target point
        """
        batch_size = len(x_start)
        # Sample continuous timesteps from uniform distribution [0, 1]
        t = torch.rand(batch_size, device=x_start.device) * self.time_scale
        # Get flow path and velocity field from start to target
        x_t, v_t = self.get_training_sample(x_start, x_target, t)
        # Model predicts velocity field
        v_pred = self.model(x_t, t, state)
        # Loss is difference between predicted and true velocity
        loss = self.loss_fn(v_pred, v_t, weights)
        return loss, x_t

    def forward(self, state, action_init=None, *args, **kwargs):
        """
        Forward pass for inference
        Always requires action_init (start point)
        """
        return self.sample(state, action_init, *args, **kwargs)


