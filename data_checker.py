try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)
import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse
from utils import my_hash, str_to_bool
import default_args
from rl_utils import GaussianPolicy, trajectory_sampler_every_step, train_policy_step, evaluate_policy, compute_bc_ql_schedule, on_policy_trajectory_collector
import psutil  # For memory monitoring
PSUTIL_AVAILABLE = True
import wandb
wandb.login(key='165a3c367eb0a61ffa11da3cc13ace2b93313835')
WANDB_AVAILABLE = True
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 / 1024  # Convert to GB

def get_gpu_memory_usage():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # Convert to GB
    return 0.0

def calculate_distribution_stats(norms_list):
    if len(norms_list) == 0:
        return {}
    norms = np.array(norms_list)
    stats = {
        'mean': np.mean(norms),
        'std': np.std(norms),
        'min': np.min(norms),
        'max': np.max(norms),
        'median': np.median(norms),
        'q25': np.percentile(norms, 25),
        'q75': np.percentile(norms, 75)
    }
    return stats

def calculate_collected_trajectory_stats(trajectory_steps_data, epoch_stats, epoch_num):
    """
    Calculate statistics for trajectory action data collected step by step during the epoch.
    
    Args:
        trajectory_steps_data: Dict with step data {step_idx: {'actions': []}}
        epoch_stats: Dictionary to store the calculated statistics
        epoch_num: Current epoch number for logging
    """
    if not trajectory_steps_data:
        return
    step_stats = {}
    for step_idx, step_data in trajectory_steps_data.items():
        actions_list = step_data
        if not actions_list:
            continue
        # Convert to numpy arrays
        actions_array = np.array(actions_list)    # Shape: (n_samples, action_dim)
        # Calculate norms for this step
        action_norms = np.linalg.norm(actions_array, axis=1)  # Shape: (n_samples,)
        # Calculate statistics for action norms
        action_norm_stats = calculate_distribution_stats(action_norms)
        
        # Calculate overall statistics for all actions across all dimensions
        all_actions_flat = actions_array.flatten()  # Shape: (n_samples * action_dim,)
        overall_action_stats = calculate_distribution_stats(all_actions_flat)
        
        # Calculate absolute value statistics for all actions
        all_actions_abs = np.abs(actions_array).flatten()  # Shape: (n_samples * action_dim,)
        action_abs_stats = calculate_distribution_stats(all_actions_abs)
        
        # Store step-wise statistics
        step_stats[f'step_{step_idx}'] = {
            'action_norm_stats': action_norm_stats,
            'overall_action_stats': overall_action_stats,
            'action_abs_stats': action_abs_stats,
            'action_norms': action_norms,
            'action_values': actions_array
        }
        
        # # Store action norm statistics in epoch stats
        # for stat_name, stat_value in action_norm_stats.items():
        #     dict_agg(epoch_stats, f'traj_step_{step_idx}_action_norm_{stat_name}', [stat_value])
        
        # # Store overall action statistics in epoch stats
        # for stat_name, stat_value in overall_action_stats.items():
        #     dict_agg(epoch_stats, f'traj_step_{step_idx}_overall_action_{stat_name}', [stat_value])
    
    # Print trajectory step statistics summary
    print(f"Epoch {epoch_num} Trajectory Step Statistics:")
    sorted_steps = sorted(step_stats.keys(), key=lambda x: int(x.split('_')[1]))
    for step_key in sorted_steps:
        step_idx = step_key.split('_')[1]
        action_norm_stats = step_stats[step_key]['action_norm_stats']
        overall_action_stats = step_stats[step_key]['overall_action_stats']
        action_abs_stats = step_stats[step_key]['action_abs_stats']
        print(f"Step {step_idx}: Action_norm(mean={action_norm_stats['mean']:.4f}, std={action_norm_stats['std']:.4f}, median={action_norm_stats['median']:.4f}, range=[{action_norm_stats['min']:.4f}, {action_norm_stats['max']:.4f}]) | Overall_action(mean={overall_action_stats['mean']:.4f}, std={overall_action_stats['std']:.4f}, median={overall_action_stats['median']:.4f}, range=[{overall_action_stats['min']:.4f}, {overall_action_stats['max']:.4f}]) | Action_abs(mean={action_abs_stats['mean']:.4f}, std={action_abs_stats['std']:.4f}, median={action_abs_stats['median']:.4f}, range=[{action_abs_stats['min']:.4f}, {action_abs_stats['max']:.4f}])")
    
    return step_stats

def calculate_norm_distribution_stats(y_init_norms_epoch, y_target_norms_epoch, y_distance_norms_epoch, epoch_stats, epoch_num):

    if len(y_init_norms_epoch) == 0:
        return {}
    
    # Calculate distribution statistics for each norm type
    y_init_stats = calculate_distribution_stats(y_init_norms_epoch)
    y_target_stats = calculate_distribution_stats(y_target_norms_epoch)
    y_distance_stats = calculate_distribution_stats(y_distance_norms_epoch)

    # Print norm distribution summary
    print(f"Epoch {epoch_num} Norm Distributions:")
    print(f"  Y_init:    mean={y_init_stats['mean']:.4f}, std={y_init_stats['std']:.4f}, median={y_init_stats['median']:.4f}, range=[{y_init_stats['min']:.4f}, {y_init_stats['max']:.4f}]")
    print(f"  Y_target:  mean={y_target_stats['mean']:.4f}, std={y_target_stats['std']:.4f}, median={y_target_stats['median']:.4f}, range=[{y_target_stats['min']:.4f}, {y_target_stats['max']:.4f}]")
    print(f"  Distance:  mean={y_distance_stats['mean']:.4f}, std={y_distance_stats['std']:.4f}, median={y_distance_stats['median']:.4f}, range=[{y_distance_stats['min']:.4f}, {y_distance_stats['max']:.4f}]")
    
    return {
        'y_init_stats': y_init_stats,
        'y_target_stats': y_target_stats,
        'y_distance_stats': y_distance_stats
    }

def main():
    parser = argparse.ArgumentParser(description='DC3')
    # Problem arguments
    parser.add_argument('--probType', type=str, default='simple', choices=['simple', 'nonconvex', 'acopf57'], help='problem type')
    parser.add_argument('--simpleVar', type=int, help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int, help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int, help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int, help='total number of datapoints for simple problem')
    parser.add_argument('--nonconvexVar', type=int, help='number of decision vars for nonconvex problem')
    parser.add_argument('--nonconvexIneq', type=int, help='number of inequality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEq', type=int, help='number of equality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEx', type=int, help='total number of datapoints for nonconvex problem')
    # Training arguments
    parser.add_argument('--epochs', type=int, default=400, help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int, help='training batch size')
    parser.add_argument('--lr', type=float, help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int, help='hidden layer size for neural network')
    parser.add_argument('--softWeight', type=float, help='total weight given to constraint violations in loss')
    parser.add_argument('--softWeightEqFrac', type=float, help='fraction of weight given to equality constraints (vs. inequality constraints) in loss')
    parser.add_argument('--useCompl', type=str_to_bool, help='whether to use completion')
    # Correction arguments(deprecated)
    parser.add_argument('--useTrainCorr', type=str_to_bool, help='whether to use correction during training')
    parser.add_argument('--useTestCorr', type=str_to_bool, help='whether to use correction during testing')
    parser.add_argument('--corrMode', choices=['partial', 'full'], help='employ DC3 correction (partial) or naive correction (full)')
    parser.add_argument('--corrTrainSteps', type=int, help='number of correction steps during training')
    parser.add_argument('--corrTestMaxSteps', type=int, help='max number of correction steps during testing')
    parser.add_argument('--corrEps', type=float, help='correction procedure tolerance')
    parser.add_argument('--corrLr', type=float, help='learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float, help='momentum for correction procedure')
    # Save arguments
    parser.add_argument('--saveAllStats', type=str_to_bool, help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int, help='how frequently (in terms of number of epochs) to save stats to file')
    # Data format arguments
    parser.add_argument('--useMultiParam', type=str_to_bool, default=True, help='whether to use multi-parameter dataset format')
    parser.add_argument('--useSanity', type=str_to_bool, default=False, help='whether to use sanity experiment dataset (same parameters, different X)')
    # Wandb arguments
    parser.add_argument('--useWandb', type=str_to_bool, default=False, help='whether to use wandb logging')
    parser.add_argument('--wandbProject', type=str, default='dc3-experiments', help='wandb project name')
    parser.add_argument('--wandbRunName', type=str, default=None, help='wandb run name (auto-generated if not provided)')
    # RL-specific arguments
    parser.add_argument('--rlLr', type=float, default=1e-4, help='learning rate for RL policy network')
    parser.add_argument('--episodeLength', type=int, default=5, help='number of steps per RL episode')
    parser.add_argument('--rlThreshold', type=float, default=1.0, help='threshold for constraint violations to start RL training')
    parser.add_argument('--alpha', type=float, default=0.7, help='weight for BC loss in hybrid policy loss')
    parser.add_argument('--beta', type=float, default=0.3, help='weight for RA loss in hybrid policy loss')
    parser.add_argument('--lambda_temp', type=float, default=1.0, help='temperature parameter for RA loss')
    parser.add_argument('--policySteps', type=int, default=5, help='number of policy update steps per epoch')
    parser.add_argument('--bufferSize', type=int, default=200000, help='size of replay buffer for RL training')
    parser.add_argument('--bufferStrategy', type=str, default='fifo', choices=['fifo', 'random'], help='buffer management strategy: fifo (first-in-first-out) or random (random removal)')
    parser.add_argument('--scheduleBC', type=str_to_bool, default=True, help='whether to use BC/QL schedule (start with pure BC, gradually add QL)')
    parser.add_argument('--scheduleBCEpochs', type=int, default=200, help='number of epochs for pure BC before starting QL schedule')
    parser.add_argument('--scheduleBCTransition', type=int, default=300, help='number of epochs to transition from BC-only to final BC/QL ratio')
    # On-policy trajectory arguments
    parser.add_argument('--onPolicyRatio', type=float, default=0.0, help='ratio of on-policy data in training batch (0.0=off-policy only, 1.0=on-policy only)')
    # Tracking arguments
    parser.add_argument('--trackStats', type=str_to_bool, default=True, help='whether to track norm distributions and step-wise trajectory statistics')
    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.method_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)
    time_str = time.strftime("%Y%b%d-%H%M%S", time.localtime())

    # Initialize wandb if requested
    if args.get('useWandb', False) and WANDB_AVAILABLE:
        run_name = args.get('wandbRunName')
        if run_name is None:
            # Auto-generate run name based on experiment type
            if args.get('useSanity', False):
                run_name = f"data-checker-{args['probType']}-{time_str}"
            else:
                run_name = f"regular-rl-on&off-data-schedule-std-neg5-2-ascorr-{args['probType']}-{time_str}"
        wandb.init(
            project=args.get('wandbProject', 'dc3-experiments'),
            name=run_name,
            config=args
        )
        print(f"Wandb initialized: {wandb.run.name}")
    elif args.get('useWandb', False) and not WANDB_AVAILABLE:
        print("WARNING: wandb requested but not available. Continuing without wandb logging.")

    setproctitle('DC3-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        if args.get('useSanity', False):
            # Use sanity experiment dataset
            filepath = os.path.join('datasets', 'simple', "random_simple_multi_param_sanity_dataset_var{}_ineq{}_eq{}_ex{}".format(
                args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
        elif args.get('useMultiParam', False):
            filepath = os.path.join('datasets', 'simple', "random_simple_multi_param_dataset_var{}_ineq{}_eq{}_ex{}".format(
                args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
        else:
            filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
                args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    elif prob_type == 'acopf57':
        filepath = os.path.join('datasets', 'acopf', 'acopf57_dataset')
    else:
        raise NotImplementedError

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE

    sanity_suffix = '_sanity' if args.get('useSanity', False) else '_regular'
    save_dir = os.path.join('results', str(data), 'method', my_hash(str(sorted(list(args.items())))),
        time_str + sanity_suffix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
    
    # Run method
    train_net(data, args, save_dir)


def train_net(data, args, save_dir):
    nepochs = args['epochs']
    batch_size = args['batchSize']

    train_dataset = TensorDataset(data.trainX, data.trainY)
    # train_dataset = TensorDataset(data.validX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Initialize networks and policy network
    initializer_net = NNSolver(data, args)  # Network 1: Initializer (supervised)
    policy_net = GaussianPolicy(data, args, hidden_dim=args['hiddenSize'])
    
    initializer_net.to(DEVICE)
    policy_net.to(DEVICE)
    
    # Optimizers
    initializer_opt = optim.Adam(initializer_net.parameters(), lr=args['lr'])
    policy_opt = optim.Adam(policy_net.parameters(), lr=args['rlLr'])
    
    # Initialize flag and buffer
    rl_enabled = False
    initial_model_good = False  # Flag to track if initial model is performing well enough
    buffer = []
    buffer_size_limit = args['bufferSize']
    buffer_strategy = args.get('bufferStrategy', 'fifo')
    on_policy_buffer = []
    on_policy_buffer_size_limit = args['bufferSize']
    on_policy_ratio = args.get('onPolicyRatio', 0.5)
    # Print buffer information
    print(f"Using buffers with max size {buffer_size_limit}, strategy: {buffer_strategy}")
    print(f"On-policy ratio: {on_policy_ratio:.2f}")
    if buffer_strategy == 'fifo':
        print("FIFO strategy: removes oldest data first (more memory efficient)")
    else:
        print("Random strategy: keeps random subset of data (may cause memory leaks)")
    
    # Track memory usage for leak detection
    initial_memory = get_memory_usage() if PSUTIL_AVAILABLE else 0
    print(f"Initial memory usage: {initial_memory:.2f} GB")
    
    stats = {}
    # Variables to store the last epoch data
    last_epoch_data = {
        'y_init': [],
        'y_target': [],
        # 'trajectory_states': [],
        'trajectory_actions': [],
        'epoch_num': -1
    }
    
    for i in range(nepochs):
        epoch_stats = {}
        # Initialize norm tracking for this epoch
        y_init_norms_epoch = []
        y_target_norms_epoch = []
        y_distance_norms_epoch = []
        
        # Initialize trajectory step-wise collection for this epoch
        trajectory_steps_data = {}  # Will store data for each step: {step_idx: {'actions': []}}
        n_steps = args.get('episodeLength', 5)  # Get number of steps from args
        
        # Variables to collect data for this epoch
        epoch_y_init = []
        epoch_y_target = []
        # epoch_trajectory_states = []
        epoch_trajectory_actions = []
        
        # Get valid loss for initializer
        initializer_net.eval()
        policy_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_net(data, Xvalid, initializer_net, args, 'valid', epoch_stats, policy_net, rl_enabled)
        # Get test loss for initializer
        for Xtest in test_loader:
            Xtest = Xtest[0].to(DEVICE)
            eval_net(data, Xtest, initializer_net, args, 'test', epoch_stats, policy_net, rl_enabled)
        # Check if initial model is good enough (based on constraint violations from current evaluation)
        if not initial_model_good and 'valid_initial_ineq_num_viol_0' in epoch_stats:
            # Consider model good if constraint violations are below threshold
            if np.mean(epoch_stats['valid_initial_ineq_num_viol_0']) <= 0:
                initial_model_good = True
                print(f"Initial model is now good enough at epoch {i} (constraint violations: {np.mean(epoch_stats['valid_initial_ineq_num_viol_0']):.3f} <= 0)")
        
        # Training loop - only train initializer if model is not good enough yet
        initializer_net.train()
        policy_net.train()
        if not initial_model_good:
            for Xtrain, Ytrain in train_loader:
                Xtrain = Xtrain.to(DEVICE)
                Ytrain = Ytrain.to(DEVICE)
                start_time = time.time()
                # Step 1: Train initializer network (supervised learning)
                initializer_opt.zero_grad()
                Yhat_train = initializer_net(Xtrain)
                train_loss = loss_w_soft_penalty(data, Xtrain, Yhat_train, args)
                train_loss.sum().backward()
                initializer_opt.step()
                train_time = time.time() - start_time
                # Record statistics
                dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
                dict_agg(epoch_stats, 'train_time', train_time, op='sum')
        else:
            dict_agg(epoch_stats, 'train_loss', np.array([0.0]*8334))
            dict_agg(epoch_stats, 'train_time', 0.0, op='sum')
            
        # Create base log message
        log_msg = 'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}'.format(
            i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_initial_eval']), 
            np.mean(epoch_stats['valid_initial_ineq_max']), np.mean(epoch_stats['valid_initial_ineq_mean']), 
            np.mean(epoch_stats['valid_initial_ineq_num_viol_0']), np.mean(epoch_stats['valid_initial_eq_max']))
        
        # Check condition to enable RL training (after evaluation step)
        if not rl_enabled and 'valid_initial_ineq_num_viol_0' in epoch_stats and np.mean(epoch_stats['valid_initial_ineq_num_viol_0']) <= args['rlThreshold']:
            print(f"Starting RL training at epoch {i} (constraint violations: {np.mean(epoch_stats['valid_initial_ineq_num_viol_0']):.3f} <= {args['rlThreshold']})")
            rl_enabled = True
        
        # RL training step (if enabled)
        if rl_enabled and not initial_model_good:
            # Step 1: Generate successful trajectories
            initializer_net.eval()
            policy_net.eval()
            
            for Xtrain, Ytrain in train_loader:
                Xtrain = Xtrain.to(DEVICE)
                Ytrain = Ytrain.to(DEVICE)
                
                with torch.no_grad():
                    Y_init = initializer_net(Xtrain)  # Get initial Y from initializer
                    Y_target = Ytrain  # Get target Y from dataset
                    # Collect norms for distribution analysis
                    # Calculate norms for each sample in the batch
                    y_init_norms_batch = torch.norm(Y_init, dim=1).cpu().numpy()
                    y_target_norms_batch = torch.norm(Y_target, dim=1).cpu().numpy()
                    y_distance_norms_batch = torch.norm(Y_init - Y_target, dim=1).cpu().numpy()
                    y_init_norms_epoch.extend(y_init_norms_batch)
                    y_target_norms_epoch.extend(y_target_norms_batch)
                    y_distance_norms_epoch.extend(y_distance_norms_batch)
                    
                    # Collect data for saving
                    epoch_y_init.append(Y_init.cpu().numpy())
                    epoch_y_target.append(Y_target.cpu().numpy())
                    
                # Generate trajectory data with step-wise format
                trajectory_states, trajectory_actions = trajectory_sampler_every_step(Y_init, Y_target, data, Xtrain, args)
                # Collect trajectory data step by step for epoch-wise statistics
                n_trajectories, n_steps_actual, action_dim = trajectory_actions.shape
                for step in range(n_steps_actual):
                    if step not in trajectory_steps_data:
                        trajectory_steps_data[step] = []
                    # Collect actions for this step across all trajectories in this batch
                    step_actions = trajectory_actions[:, step, :].cpu().numpy()    # Shape: (n_trajectories, action_dim)
                    trajectory_steps_data[step].extend(step_actions)
                
                # Collect trajectory data for saving
                # epoch_trajectory_states.append(trajectory_states.cpu().numpy())
                epoch_trajectory_actions.append(trajectory_actions.cpu().numpy())
                
        print(log_msg)
        
        # Calculate and log norm distribution statistics
        if args.get('trackStats', True):
            calculate_norm_distribution_stats(y_init_norms_epoch, y_target_norms_epoch, y_distance_norms_epoch, epoch_stats, i)
            # Calculate trajectory step-wise statistics from collected data
            calculate_collected_trajectory_stats(trajectory_steps_data, epoch_stats, i)
        
        # Update last epoch data if we have collected data
        if len(epoch_y_init) > 0:
            last_epoch_data = {
                'y_init': np.concatenate(epoch_y_init, axis=0),  # Concatenate all batches
                'y_target': np.concatenate(epoch_y_target, axis=0),  # Concatenate all batches
                # 'trajectory_states': np.concatenate(epoch_trajectory_states, axis=0),  # Concatenate all batches
                'trajectory_actions': np.concatenate(epoch_trajectory_actions, axis=0),  # Concatenate all batches
                'epoch_num': i
            }
        
        # Memory monitoring every epoch
        if PSUTIL_AVAILABLE:
            memory_gb = get_memory_usage()
            gpu_memory_gb = get_gpu_memory_usage()
            buffer_info = f"Off-policy buffer: {len(buffer)}, On-policy buffer: {len(on_policy_buffer)}"
            print(f"Epoch {i} - RAM: {memory_gb:.2f} GB, GPU: {gpu_memory_gb:.2f} GB, {buffer_info}")
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Basic garbage collection (FIFO strategy handles most memory management)
        import gc
        gc.collect()

        # Log to wandb if available
        if args.get('useWandb', False) and WANDB_AVAILABLE:
            wandb_log_dict = {
                'epoch': i,
                'train_loss': np.mean(epoch_stats['train_loss']),
                'valid_initial_eval': np.mean(epoch_stats['valid_initial_eval']),
                'valid_initial_ineq_max': np.mean(epoch_stats['valid_initial_ineq_max']),
                'valid_initial_ineq_mean': np.mean(epoch_stats['valid_initial_ineq_mean']),
                'valid_initial_ineq_num_viol_0': np.mean(epoch_stats['valid_initial_ineq_num_viol_0']),
                'valid_initial_eq_max': np.mean(epoch_stats['valid_initial_eq_max']),
                'valid_initial_eq_mean': np.mean(epoch_stats['valid_initial_eq_mean']),
                'valid_initial_time': np.mean(epoch_stats['valid_initial_time']),
                'train_time': np.mean(epoch_stats['train_time']),
                'rl_enabled': float(rl_enabled),
                'off_policy_buffer_size': len(buffer),
                'on_policy_buffer_size': len(on_policy_buffer),
                'on_policy_ratio': on_policy_ratio
            }
            wandb.log(wandb_log_dict)
    
    # Save the last epoch data
    if last_epoch_data['epoch_num'] >= 0:
        save_path = os.path.join(save_dir, 'last_epoch_data.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(last_epoch_data, f)
        print(f"Saved last epoch data (epoch {last_epoch_data['epoch_num']}) to {save_path}")
        print(f"Data shapes: Y_init {last_epoch_data['y_init'].shape}, Y_target {last_epoch_data['y_target'].shape}")
        print(f"Trajectory shapes: Actions {last_epoch_data['trajectory_actions'].shape}")
    
    # Finish wandb run
    if args.get('useWandb', False) and WANDB_AVAILABLE:
        wandb.finish()
        print("Wandb run finished")
    
    return initializer_net, policy_net, stats

# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value

# Modifies stats in place
def eval_net(data, X, solver_net, args, prefix, stats, policy_net=None, rl_enabled=False):
    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y_initial = solver_net(X)
    end_time = time.time()

    # Record initial solution metrics
    dict_agg(stats, make_prefix('initial_time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('initial_loss'), loss_w_soft_penalty(data, X, Y_initial, args).detach().cpu().numpy())
    if hasattr(data, 'data_all'):
        # This is a SimpleProblemMultiParam dataset
        dict_agg(stats, make_prefix('initial_eval'), data.obj_fn(Y_initial, X).detach().cpu().numpy())
    else:
        # This is a regular SimpleProblem dataset
        dict_agg(stats, make_prefix('initial_eval'), data.obj_fn(Y_initial).detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_ineq_max'), torch.max(data.ineq_dist(X, Y_initial), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_ineq_mean'), torch.mean(data.ineq_dist(X, Y_initial), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Y_initial) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Y_initial) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Y_initial) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Y_initial)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Y_initial)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_eq_num_viol_0'),
             torch.sum(torch.abs(data.eq_resid(X, Y_initial)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_eq_num_viol_1'),
             torch.sum(torch.abs(data.eq_resid(X, Y_initial)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_eq_num_viol_2'),
             torch.sum(torch.abs(data.eq_resid(X, Y_initial)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    
    return stats

def loss_w_soft_penalty(data, X, Y, args):
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


######### Models
class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        input_dim = data.xdim
        layer_sizes = [input_dim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        output_dim = data.ydim - data.nknowns
        if self._args['useCompl']:
            layers += [nn.Linear(layer_sizes[-1], output_dim - data.neq)]
        else:
            layers += [nn.Linear(layer_sizes[-1], output_dim)]
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if self._args['useCompl']:
            if 'acopf' in self._args['probType']:
                out = nn.Sigmoid()(out)   # used to interpolate between max and min values
            out_full = self._data.complete_partial_parallel(x, out)
            # return self._data.complete_partial_parallel(x, out)
        else:
            out_full = out

        return out_full


if __name__=='__main__':
    main()
