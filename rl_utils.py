import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

def loss_w_soft_penalty(data, X, Y, args):
    """state value function for RL"""
    if hasattr(data, 'data_all'):
        # This is a SimpleProblemMultiParam dataset
        obj_cost = data.obj_fn(Y, X)
    else:
        # This is a regular SimpleProblem dataset
        obj_cost = data.obj_fn(Y)
    ineq_dist = data.ineq_dist(X, Y)
    ineq_cost = torch.norm(ineq_dist, dim=1)
    eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
    return obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + \
            args['softWeight'] * args['softWeightEqFrac'] * eq_cost

class GaussianPolicy(nn.Module):
    """Gaussian Policy for continuous action space"""
    def __init__(self, data, args, hidden_dim=200):
        super().__init__()
        self._data = data
        self._args = args
        self.partial_action_dim = data.ydim - data.neq
        
        # Input dimension: state_dim (x + y)
        self.state_dim = data.xdim + data.ydim
        
        # Output dimension: partial variables only
        if args['useCompl']:
            self.action_dim = data.ydim - data.neq  # Partial variables only
        else:
            self.action_dim = data.ydim  # Full variables
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        # Policy heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.action_dim)
        
        # Initialize weights
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
        nn.init.kaiming_normal_(self.mean_head.weight)
        nn.init.kaiming_normal_(self.log_std_head.weight)
        # Initialize log_std to small values
        self.log_std_head.bias.data.fill_(-2.0)
    
    def forward(self, s):
        """
        Forward pass to get action parameters
        Args:
            s: state tensor of shape (batch_size, state_dim) - concatenated [x, y]
        Returns:
            mu: mean of action distribution (partial actions)
            log_std: log standard deviation of action distribution (partial actions)
        """
        features = self.feature_net(s)
        mu = self.mean_head(features)
        log_std = self.log_std_head(features)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -5.0, 2.0)  # std ∈ [0.0067, 7.4]
        return mu, log_std
    
    def sample_action(self, s):
        """
        Sample action from policy
        Args:
            s: state tensor of shape (batch_size, state_dim) - concatenated [x, y]
        Returns:
            action: action tensor of shape (batch_size, action_dim)
        """
        mu, log_std = self.forward(s)
        std = torch.exp(log_std)
        # Create normal distribution
        normal = Normal(mu, std)
        # Sample action
        action = normal.rsample()  # Use rsample for reparameterization trick
        action = torch.clamp(action, -0.3, 0.2)
        return action
    
    def get_log_prob(self, s, a):
        """
        Get log probability of action given state
        Args:
            s: state tensor of shape (batch_size, state_dim)
            a: action tensor of shape (batch_size, action_dim)
        Returns:
            log_prob: log probability of the action
        """
        mu, log_std = self.forward(s)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        log_prob = normal.log_prob(a)
        # Sum log probabilities across action dimensions
        log_prob = log_prob.sum(dim=-1)
        return log_prob

    def state_transition(self, x, y_current, action):
        """
        state transition function for RL
        Args:
            x: 输入 (batch_size, xdim)
            y_current: 当前Y (batch_size, ydim) 
            action: 部分action (batch_size, action_dim)
        Returns:
            y_new: 新的完整Y (batch_size, ydim)
        """
        if self._args['useCompl']:
            y_new_partial = y_current[:, :self.partial_action_dim] + action
            y_new = self._data.complete_partial_parallel(x, y_new_partial)
        else:
            y_new = y_current + action
        return y_new

def Q_s_a(policy, data, s, a, args):
    """
    Compute negative Q-value (to be maximized) for state-action pairs
    Args:
        policy: GaussianPolicy object
        data: Dataset object
        s: state tensor of shape (batch_size, x_dim + y_dim)
        a: partial action tensor of shape (batch_size, action_dim)
        args: Arguments
    """
    x_dim = data.xdim
    x = s[:, :x_dim]
    y_current = s[:, x_dim:]
    
    # Use policy's state_transition to get new Y
    y_new = policy.state_transition(x, y_current, a)
    
    q_s_a = loss_w_soft_penalty(data, x, y_new, args)
    return q_s_a

# def compute_advantage_simple(data, s, a, args):
#     """Simple advantage estimation based on loss improvement (Option B)"""
#     # Split state back to x and y
#     x_dim = data.xdim
#     x = s[:, :x_dim]
#     y = s[:, x_dim:]
#     # Apply action (update)
#     y_new = y + a
#     # Compute advantage as negative loss difference
#     loss_old = loss_w_soft_penalty(data, x, y, args)
#     loss_new = loss_w_soft_penalty(data, x, y_new, args)
#     advantage = -(loss_new - loss_old)  # Negative because we want to minimize loss
#     return advantage

def trajectory_sampler(start_point, end_point, data, batch, args, n_trajectory_per_data=1):
    """
    Generate multiple trajectory data in (s, a) format.
    Args:
        start_point: tensor of shape (batch_size, ydim) - model output (y_0)
        end_point: tensor of shape (batch_size, ydim) - saved answer (y_1)
        data: Dataset object
        batch: Input batch for loss computation
        args: Arguments for loss computation
        n_steps: number of steps to sample between start and end
        n_trajectory_per_data: number of trajectories per data point
    Returns:
        states: tensor of shape (batch_size * n_trajectory_per_data * n_steps, xdim + ydim) - concatenated [x, y]
        actions: tensor of shape (batch_size * n_trajectory_per_data * n_steps, ydim) - action (delta_y)
    """
    n_steps = args['episodeLength']
    batch_size = start_point.shape[0]
    x_dim = data.xdim
    y_dim = data.ydim
    
    # Calculate total distance from start to end
    total_distance = end_point - start_point  # Shape: (batch_size, ydim)
    
    # Total number of trajectories
    total_trajectories = batch_size * n_trajectory_per_data
    
    # Batch sample random step lengths for all trajectories at once
    # Shape: (total_trajectories, n_steps, y_dim)
    step_lengths_raw = torch.rand(total_trajectories, n_steps, y_dim, device=start_point.device)
    
    # Normalize step lengths so they sum to total distance for each trajectory and dimension
    # Shape: (total_trajectories, n_steps, y_dim)
    step_lengths = step_lengths_raw / step_lengths_raw.sum(dim=1, keepdim=True)  # Normalize to sum to 1
    
    # Sort step lengths along the steps dimension (dim=1) in descending order
    # This ensures larger steps come first (early in trajectory) and smaller steps come later (near target)
    step_lengths_sorted, _ = torch.sort(step_lengths, dim=1, descending=True)
    
    # Expand total_distance to match trajectory dimensions
    # Shape: (batch_size, 1, y_dim) -> (total_trajectories, 1, y_dim)
    total_distance_expanded = total_distance.unsqueeze(1).repeat(1, n_trajectory_per_data, 1)
    total_distance_expanded = total_distance_expanded.view(total_trajectories, 1, y_dim)
    
    # Scale step lengths by total distance
    # step_lengths = step_lengths * total_distance_expanded  # Shape: (total_trajectories, n_steps, y_dim)
    step_lengths = step_lengths_sorted * total_distance_expanded  # Shape: (total_trajectories, n_steps, y_dim)

    # Compute actions (step lengths) and clamp them first
    # Shape: (total_trajectories, n_steps, y_dim)
    actions = step_lengths.clone()
    # Clamp actions to [-1, 1]
    actions = torch.clamp(actions, -0.3, 0.2)
    
    # Expand start points to match trajectory dimensions
    # Shape: (batch_size, y_dim) -> (total_trajectories, y_dim)
    start_points_expanded = start_point.repeat(1, n_trajectory_per_data).view(total_trajectories, y_dim)
    
    # Compute trajectory positions step by step using clamped actions
    # Shape: (total_trajectories, n_steps, y_dim)
    trajectory_positions = torch.zeros(total_trajectories, n_steps, y_dim, device=start_point.device)
    
    # First position is the start point
    trajectory_positions[:, 0, :] = start_points_expanded
    
    # Compute subsequent positions by adding clamped actions
    for step_idx in range(1, n_steps):
        trajectory_positions[:, step_idx, :] = trajectory_positions[:, step_idx-1, :] + actions[:, step_idx-1, :]
    
    # Expand batch data to match trajectory dimensions
    # Shape: (batch_size, x_dim) -> (total_trajectories, x_dim)
    batch_expanded = batch.repeat(1, n_trajectory_per_data).view(total_trajectories, x_dim)
    
    # Reshape for final output
    # Shape: (total_trajectories * n_steps, x_dim + y_dim)
    states = torch.cat([
        batch_expanded.unsqueeze(1).expand(-1, n_steps, -1).reshape(-1, x_dim),
        trajectory_positions.reshape(-1, y_dim)
    ], dim=1)
    
    # Shape: (total_trajectories * n_steps, y_dim)
    actions = actions.reshape(-1, y_dim)
    
    return states, actions

def trajectory_sampler_every_step(start_point, end_point, data, batch, args, n_trajectory_per_data=1):
    """
    Generate multiple trajectory data in (s, a) format.
    Args:
        start_point: tensor of shape (batch_size, ydim) - model output (y_0)
        end_point: tensor of shape (batch_size, ydim) - saved answer (y_1)
        data: Dataset object
        batch: Input batch for loss computation
        args: Arguments for loss computation
        n_steps: number of steps to sample between start and end
        n_trajectory_per_data: number of trajectories per data point
    Returns:
        trajectory_states: tensor of shape (batch_size * n_trajectory_per_data, n_steps, xdim + ydim) - concatenated [x, y] with preserved step dimension
        trajectory_actions: tensor of shape (batch_size * n_trajectory_per_data, n_steps, ydim) - action (delta_y) with preserved step dimension
    """
    n_steps = args['episodeLength']
    batch_size = start_point.shape[0]
    x_dim = data.xdim
    y_dim = data.ydim
    
    # Calculate total distance from start to end
    total_distance = end_point - start_point  # Shape: (batch_size, ydim)
    
    # Total number of trajectories
    total_trajectories = batch_size * n_trajectory_per_data
    
    # Batch sample random step lengths for all trajectories at once
    # Shape: (total_trajectories, n_steps, y_dim)
    step_lengths_raw = torch.rand(total_trajectories, n_steps, y_dim, device=start_point.device)
    
    # Normalize step lengths so they sum to total distance for each trajectory and dimension
    # Shape: (total_trajectories, n_steps, y_dim)
    step_lengths = step_lengths_raw / step_lengths_raw.sum(dim=1, keepdim=True)  # Normalize to sum to 1
    
    # Sort step lengths along the steps dimension (dim=1) in descending order
    # This ensures larger steps come first (early in trajectory) and smaller steps come later (near target)
    step_lengths_sorted, _ = torch.sort(step_lengths, dim=1, descending=True)
    
    # Expand total_distance to match trajectory dimensions
    # Shape: (batch_size, 1, y_dim) -> (total_trajectories, 1, y_dim)
    total_distance_expanded = total_distance.unsqueeze(1).repeat(1, n_trajectory_per_data, 1)
    total_distance_expanded = total_distance_expanded.view(total_trajectories, 1, y_dim)
    
    # Scale step lengths by total distance
    # step_lengths = step_lengths * total_distance_expanded  # Shape: (total_trajectories, n_steps, y_dim)
    step_lengths = step_lengths_sorted * total_distance_expanded  # Shape: (total_trajectories, n_steps, y_dim)

    # Compute actions (step lengths) and clamp them first
    # Shape: (total_trajectories, n_steps, y_dim)
    actions = step_lengths.clone()
    # Clamp actions to [-1, 1]
    actions = torch.clamp(actions, -0.3, 0.2)
    
    # Expand start points to match trajectory dimensions
    # Shape: (batch_size, y_dim) -> (total_trajectories, y_dim)
    start_points_expanded = start_point.repeat(1, n_trajectory_per_data).view(total_trajectories, y_dim)
    
    # Compute trajectory positions step by step using clamped actions
    # Shape: (total_trajectories, n_steps, y_dim)
    trajectory_positions = torch.zeros(total_trajectories, n_steps, y_dim, device=start_point.device)
    
    # First position is the start point
    trajectory_positions[:, 0, :] = start_points_expanded
    
    # Compute subsequent positions by adding clamped actions
    for step_idx in range(1, n_steps):
        trajectory_positions[:, step_idx, :] = trajectory_positions[:, step_idx-1, :] + actions[:, step_idx-1, :]
    
    # Expand batch data to match trajectory dimensions
    # Shape: (batch_size, x_dim) -> (total_trajectories, x_dim)
    batch_expanded = batch.repeat(1, n_trajectory_per_data).view(total_trajectories, x_dim)
    
    # Prepare output with preserved step dimensions
    # Shape: (total_trajectories, n_steps, x_dim + y_dim)
    trajectory_states = torch.cat([
        batch_expanded.unsqueeze(1).expand(-1, n_steps, -1),  # (total_trajectories, n_steps, x_dim)
        trajectory_positions  # (total_trajectories, n_steps, y_dim)
    ], dim=2)
    
    # Shape: (total_trajectories, n_steps, y_dim)
    trajectory_actions = actions
    
    return trajectory_states, trajectory_actions

def on_policy_trajectory_collector(start_point, policy_net, data, batch, args, n_trajectory_per_data=1):
    """
    Generate on-policy trajectory data using the current policy network.
    
    Args:
        start_point: tensor of shape (batch_size, ydim) - initial Y from initializer network
        policy_net: GaussianPolicy network for sampling actions
        data: Dataset object
        batch: Input batch (X) for state construction
        args: Arguments containing episodeLength
        n_trajectory_per_data: number of trajectories per data point
    
    Returns:
        states: tensor of shape (batch_size * n_trajectory_per_data * n_steps, xdim + ydim) - concatenated [x, y]
        actions: tensor of shape (batch_size * n_trajectory_per_data * n_steps, ydim) - policy-sampled actions
    """
    n_steps = args['episodeLength']
    batch_size = start_point.shape[0]
    x_dim = data.xdim
    y_dim = data.ydim
    
    # Total number of trajectories
    total_trajectories = batch_size * n_trajectory_per_data
    
    # Expand start points to match trajectory dimensions
    # Shape: (batch_size, y_dim) -> (total_trajectories, y_dim)
    start_points_expanded = start_point.repeat(1, n_trajectory_per_data).view(total_trajectories, y_dim)
    
    # Expand batch data to match trajectory dimensions
    # Shape: (batch_size, x_dim) -> (total_trajectories, x_dim)
    batch_expanded = batch.repeat(1, n_trajectory_per_data).view(total_trajectories, x_dim)
    
    # Initialize trajectory storage
    # Shape: (total_trajectories, n_steps, y_dim)
    trajectory_positions = torch.zeros(total_trajectories, n_steps, y_dim, device=start_point.device)
    # Shape: (total_trajectories, n_steps, y_dim)
    trajectory_actions = torch.zeros(total_trajectories, n_steps, y_dim, device=start_point.device)
    
    # First position is the start point
    trajectory_positions[:, 0, :] = start_points_expanded
    
    # Generate trajectory using policy network
    with torch.no_grad():
        for step_idx in range(n_steps):
            # Current state: concatenate X and current Y
            # Shape: (total_trajectories, x_dim + y_dim)
            current_state = torch.cat([batch_expanded, trajectory_positions[:, step_idx, :]], dim=1)
            
            # Sample action from policy
            # Shape: (total_trajectories, y_dim)
            # action, _ = policy_net.sample_action(current_state)
            action = policy_net.sample_action(current_state)
            trajectory_actions[:, step_idx, :] = action
            
            # Update position for next step (if not the last step)
            if step_idx < n_steps - 1:
                trajectory_positions[:, step_idx + 1, :] = trajectory_positions[:, step_idx, :] + action
    
    # Reshape for final output
    # Shape: (total_trajectories * n_steps, x_dim + y_dim)
    states = torch.cat([
        batch_expanded.unsqueeze(1).expand(-1, n_steps, -1).reshape(-1, x_dim),
        trajectory_positions.reshape(-1, y_dim)
    ], dim=1)
    
    # Shape: (total_trajectories * n_steps, y_dim)
    actions = trajectory_actions.reshape(-1, y_dim)
    
    return states, actions

def compute_bc_ql_schedule(epoch, args):
    """
    Compute BC/QL loss weights based on training schedule
    
    Args:
        epoch: Current epoch number
        args: Arguments containing schedule parameters
    
    Returns:
        dict with keys:
        - alpha: Current BC weight
        - beta: Current QL weight  
        - phase: Schedule phase name
        - progress: Transition progress (0-1)
    """
    if not args.get('scheduleBC', True):
        # No schedule: use fixed ratios
        return {
            'alpha': args['alpha'],
            'beta': args['beta'],
            'phase': 'Fixed',
            'progress': 1.0
        }
    
    bc_epochs = args.get('scheduleBCEpochs', 100)
    transition_epochs = args.get('scheduleBCTransition', 200)
    target_alpha = args['alpha']
    target_beta = args['beta']
    
    if epoch < bc_epochs:
        # Pure BC learning phase
        return {
            'alpha': 1.0,
            'beta': 0.0,
            'phase': 'Pure BC',
            'progress': 0.0
        }
    elif epoch < bc_epochs + transition_epochs:
        # Transition phase: gradually increase QL
        progress = (epoch - bc_epochs) / transition_epochs
        current_alpha = 1.0 - progress * (1.0 - target_alpha)
        current_beta = progress * target_beta
        
        return {
            'alpha': current_alpha,
            'beta': current_beta,
            'phase': f'Transition ({progress:.2f})',
            'progress': progress
        }
    else:
        # Final phase: use target ratios
        return {
            'alpha': target_alpha,
            'beta': target_beta,
            'phase': 'Final BC+QL',
            'progress': 1.0
        }


def hybrid_policy_loss(policy, data, states, actions, args, alpha=0.7, beta=0.3, lambda_temp=0.1):
    """
    Compute hybrid policy loss: α * L_BC + β * L_QL
    
    Args:
        policy: GaussianPolicy
        data: Dataset object
        states: tensor of shape (batch_size, state_dim)
        actions: tensor of shape (batch_size, action_dim) - partial trajectory actions
        args: Arguments for loss computation
        alpha: Weight for BC loss
        beta: Weight for Q-learning loss
        lambda_temp: Temperature parameter for Q-learning loss
    
    Returns:
        total_loss: Combined policy loss
        bc_loss: Behavioral cloning loss (MSE)
        ql_loss: Q-learning loss
    """
    # batch_size = states.shape[0]
    
    # Sample partial actions from policy
    sampled_partial_actions = policy.sample_action(states)  # Shape: (batch_size, action_dim)
    
    # Behavioral Cloning Loss: L_BC = MSE(sampled_partial_actions, trajectory_partial_actions)
    bc_loss = F.mse_loss(sampled_partial_actions, actions[:, :policy.action_dim])
    
    # Q-learning loss: minimize negative Q-values using partial actions
    ql_loss_raw = Q_s_a(policy, data, states, sampled_partial_actions, args).mean()
    ql_loss = ql_loss_raw/100.0
    
    # Combine losses
    total_loss = alpha * bc_loss + beta * ql_loss
    
    return total_loss, bc_loss, ql_loss

def train_policy_step(policy, data, states, actions, args, optimizer, 
                     alpha=0.7, beta=0.3, lambda_temp=0.1):
    """
    Single training step for policy network
    
    Returns:
        loss_info: Dictionary containing loss values
    """
    optimizer.zero_grad()
    total_loss, bc_loss, ql_loss = hybrid_policy_loss(
        policy, data, states, actions, args, alpha, beta, lambda_temp
    )
    
    total_loss.backward()
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'bc_loss': bc_loss.item(),
        'ql_loss': ql_loss.item()
    }

def evaluate_policy(policy, data, states, actions, args):
    """
    Evaluate policy performance without updating
    
    Returns:
        eval_info: Dictionary containing evaluation metrics
    """
    with torch.no_grad():
        log_probs = policy.get_log_prob(states, actions)
        
        # Compute metrics
        avg_log_prob = log_probs.mean().item()
        
        return {
            'avg_log_prob': avg_log_prob
        } 