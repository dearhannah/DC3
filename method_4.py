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
from rl_utils import GaussianPolicy, trajectory_sampler, train_policy_step, evaluate_policy, compute_bc_ql_schedule, on_policy_trajectory_collector
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

def main():
    parser = argparse.ArgumentParser(description='DC3')
    # Problem arguments
    parser.add_argument('--probType', type=str, default='simple',
        choices=['simple', 'nonconvex', 'acopf57'], help='problem type')
    parser.add_argument('--simpleVar', type=int, 
        help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int,
        help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int,
        help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int,
        help='total number of datapoints for simple problem')
    parser.add_argument('--nonconvexVar', type=int,
        help='number of decision vars for nonconvex problem')
    parser.add_argument('--nonconvexIneq', type=int,
        help='number of inequality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEq', type=int,
        help='number of equality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEx', type=int,
        help='total number of datapoints for nonconvex problem')
    # Training arguments
    parser.add_argument('--epochs', type=int,
        help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int,
        help='training batch size')
    parser.add_argument('--lr', type=float,
        help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int,
        help='hidden layer size for neural network')
    parser.add_argument('--softWeight', type=float,
        help='total weight given to constraint violations in loss')
    parser.add_argument('--softWeightEqFrac', type=float,
        help='fraction of weight given to equality constraints (vs. inequality constraints) in loss')
    parser.add_argument('--useCompl', type=str_to_bool,
        help='whether to use completion')
    parser.add_argument('--useTrainCorr', type=str_to_bool,
        help='whether to use correction during training')
    parser.add_argument('--useTestCorr', type=str_to_bool,
        help='whether to use correction during testing')
    parser.add_argument('--corrMode', choices=['partial', 'full'],
        help='employ DC3 correction (partial) or naive correction (full)')
    parser.add_argument('--corrTrainSteps', type=int,
        help='number of correction steps during training')
    parser.add_argument('--corrTestMaxSteps', type=int,
        help='max number of correction steps during testing')
    parser.add_argument('--corrEps', type=float,
        help='correction procedure tolerance')
    parser.add_argument('--corrLr', type=float,
        help='learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float,
        help='momentum for correction procedure')
    # Save arguments
    parser.add_argument('--saveAllStats', type=str_to_bool,
        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
        help='how frequently (in terms of number of epochs) to save stats to file')
    # Data format arguments
    parser.add_argument('--useMultiParam', type=str_to_bool, default=True,
        help='whether to use multi-parameter dataset format')
    parser.add_argument('--useSanity', type=str_to_bool, default=False,
        help='whether to use sanity experiment dataset (same parameters, different X)')
    # Wandb arguments
    parser.add_argument('--useWandb', type=str_to_bool, default=False,
        help='whether to use wandb logging')
    parser.add_argument('--wandbProject', type=str, default='dc3-experiments',
        help='wandb project name')
    parser.add_argument('--wandbRunName', type=str, default=None,
        help='wandb run name (auto-generated if not provided)')
    # RL-specific arguments
    parser.add_argument('--rlLr', type=float, default=1e-4,
        help='learning rate for RL policy network')
    parser.add_argument('--episodeLength', type=int, default=5,
        help='number of steps per RL episode')
    parser.add_argument('--rlThreshold', type=float, default=0.0,
        help='threshold for constraint violations to start RL training')
    parser.add_argument('--alpha', type=float, default=0.7,
        help='weight for BC loss in hybrid policy loss')
    parser.add_argument('--beta', type=float, default=0.3,
        help='weight for RA loss in hybrid policy loss')
    parser.add_argument('--lambda_temp', type=float, default=1.0,
        help='temperature parameter for RA loss')
    parser.add_argument('--policySteps', type=int, default=5,
        help='number of policy update steps per epoch')
    parser.add_argument('--bufferSize', type=int, default=200000,
        help='size of replay buffer for RL training')
    parser.add_argument('--bufferStrategy', type=str, default='fifo', choices=['fifo', 'random'],
        help='buffer management strategy: fifo (first-in-first-out) or random (random removal)')
    parser.add_argument('--scheduleBC', type=str_to_bool, default=True,
        help='whether to use BC/QL schedule (start with pure BC, gradually add QL)')
    parser.add_argument('--scheduleBCEpochs', type=int, default=200,
        help='number of epochs for pure BC before starting QL schedule')
    parser.add_argument('--scheduleBCTransition', type=int, default=300,
        help='number of epochs to transition from BC-only to final BC/QL ratio')
    # On-policy trajectory arguments
    parser.add_argument('--onPolicyRatio', type=float, default=0.0,
        help='ratio of on-policy data in training batch (0.0=off-policy only, 1.0=on-policy only)')
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
                  run_name = f"sanity-rl-small-actions-long--seperate-train-{args['probType']}-{time_str}"
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

    # Add sanity indicator to save directory
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
    state_dim = data.xdim + data.ydim  # [x, y] concatenated
    action_dim = data.ydim  # Δy dimension
    policy_net = GaussianPolicy(state_dim, action_dim, hidden_dim=args['hiddenSize'])
    
    initializer_net.to(DEVICE)
    policy_net.to(DEVICE)
    
    # Optimizers
    initializer_opt = optim.Adam(initializer_net.parameters(), lr=args['lr'])
    policy_opt = optim.Adam(policy_net.parameters(), lr=args['rlLr'])
    
    # Initialize RL flag, buffer and on-policy buffer
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
    for i in range(nepochs):
        epoch_stats = {}
        # Get valid loss for initializer
        initializer_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_net(data, Xvalid, initializer_net, args, 'valid', epoch_stats, policy_net, rl_enabled)
        # Get test loss for initializer
        initializer_net.eval()
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
        if rl_enabled:
            # Get current BC/QL schedule
            schedule = compute_bc_ql_schedule(i, args)
            current_alpha = schedule['alpha']
            current_beta = schedule['beta']
            
            # Step 1: Generate successful trajectories
            initializer_net.eval()
            policy_net.eval()
            
            for Xtrain, Ytrain in train_loader:
                Xtrain = Xtrain.to(DEVICE)
                Ytrain = Ytrain.to(DEVICE)
                
                with torch.no_grad():
                    Y_init = initializer_net(Xtrain)  # Get initial Y from initializer
                    Y_target = Ytrain  # Get target Y from dataset
                    
                # Generate trajectory data
                states, actions = trajectory_sampler(Y_init, Y_target, data, Xtrain, args)
                # Add to buffer and update buffer length
                new_data = list(zip(states.cpu().numpy(), actions.cpu().numpy()))
                buffer.extend(new_data)
                current_buffer_size = len(buffer)
                # Manage buffer size based on strategy
                if current_buffer_size > buffer_size_limit:
                    excess = current_buffer_size - buffer_size_limit
                    if args.get('bufferStrategy', 'fifo') == 'fifo':
                        # FIFO strategy: remove oldest data
                        buffer = buffer[excess:]
                        # print(f"Buffer FIFO cleaned: {current_buffer_size} -> {len(buffer)}")
                    else:  # random strategy
                        # Random strategy: keep random subset
                        indices_to_keep = np.random.choice(current_buffer_size, buffer_size_limit, replace=False)
                        indices_to_keep = np.sort(indices_to_keep)  # Sort for efficient indexing
                        # Create new buffer and explicitly delete old one
                        old_buffer = buffer
                        buffer = [old_buffer[i] for i in indices_to_keep]
                        # del old_buffer  # Explicitly delete old buffer
                        # Force garbage collection to free memory
                        import gc
                        gc.collect()
                        # print(f"Buffer random cleaned: {current_buffer_size} -> {len(buffer)}")

                # # Generate on-policy trajectory data using current policy
                # on_policy_states, on_policy_actions = on_policy_trajectory_collector(Y_init, policy_net, data, Xtrain, args)
                # # Add to on-policy buffer
                # new_on_policy_data = list(zip(on_policy_states.cpu().numpy(), on_policy_actions.cpu().numpy()))
                # on_policy_buffer.extend(new_on_policy_data)
                # current_on_policy_buffer_size = len(on_policy_buffer)
                # # Manage on-policy buffer size based on strategy
                # if current_on_policy_buffer_size > on_policy_buffer_size_limit:
                #     excess = current_on_policy_buffer_size - on_policy_buffer_size_limit
                #     if args.get('bufferStrategy', 'fifo') == 'fifo':
                #         # FIFO strategy: remove oldest data
                #         on_policy_buffer = on_policy_buffer[excess:]
                #     else:  # random strategy
                #         # Random strategy: keep random subset
                #         indices_to_keep = np.random.choice(current_on_policy_buffer_size, on_policy_buffer_size_limit, replace=False)
                #         indices_to_keep = np.sort(indices_to_keep)
                #         # Create new buffer and explicitly delete old one
                #         old_on_policy_buffer = on_policy_buffer
                #         on_policy_buffer = [old_on_policy_buffer[i] for i in indices_to_keep]
                #         # Force garbage collection to free memory
                #         import gc
                #         gc.collect()
            
            policy_net.train()  # Set back to train mode
            # Step 2: Train policy network (BC-RL)
            policy_losses = []
            current_buffer_size = len(buffer)
            current_on_policy_buffer_size = len(on_policy_buffer)
            # Determine batch composition based on on_policy_ratio
            on_policy_batch_size = int(batch_size * on_policy_ratio) 
            off_policy_batch_size = batch_size - on_policy_batch_size
            for _ in range(args['policySteps']):
                # Sample from off-policy buffer
                off_policy_indices = np.random.choice(current_buffer_size, min(off_policy_batch_size, current_buffer_size), replace=False)
                off_policy_batch_data = [buffer[i] for i in off_policy_indices]
                # Sample from on-policy buffer
                on_policy_indices = np.random.choice(current_on_policy_buffer_size, min(on_policy_batch_size, current_on_policy_buffer_size), replace=False)
                on_policy_batch_data = [on_policy_buffer[i] for i in on_policy_indices]
                # Combine batches
                batch_data = off_policy_batch_data + on_policy_batch_data
                # Convert to numpy arrays first, then to tensors (avoids slow list conversion)
                batch_states_np = np.array([item[0] for item in batch_data])
                batch_actions_np = np.array([item[1] for item in batch_data])
                batch_states = torch.tensor(batch_states_np, device=DEVICE)
                batch_actions = torch.tensor(batch_actions_np, device=DEVICE)
                    
                # Multiple policy update steps
                loss_info = train_policy_step(
                    policy_net, data, batch_states, batch_actions, args, policy_opt,
                    alpha=current_alpha, beta=current_beta, lambda_temp=args['lambda_temp']
                    )
                policy_losses.append(loss_info)
                    
                # Clean up intermediate variables
                # del batch_data, batch_states_np, batch_actions_np, batch_states, batch_actions, indices
                
            # Record policy training statistics (average if trained, zeros if not)
            dict_agg(epoch_stats, 'policy_total_loss', [np.mean([info['total_loss'] for info in policy_losses])])
            dict_agg(epoch_stats, 'policy_bc_loss', [np.mean([info['bc_loss'] for info in policy_losses])])
            dict_agg(epoch_stats, 'policy_ql_loss', [np.mean([info['ql_loss'] for info in policy_losses])])
        
            # Update log message with RL information
            # Check if RL statistics exist before accessing them
            if 'valid_eval' in epoch_stats and 'policy_total_loss' in epoch_stats:
                improvement = np.mean(epoch_stats['valid_initial_eval']) - np.mean(epoch_stats['valid_eval'])
                base_rl_msg = ' | After RL: eval {:.4f} (improvement: {:.4f}), dist {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq viol {:.1f}, eq max {:.4f}, policy loss {:.4f}'.format(
                    np.mean(epoch_stats['valid_eval']), improvement, np.mean(epoch_stats['valid_dist']), 
                    np.mean(epoch_stats['valid_ineq_max']), np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']), 
                    np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['policy_total_loss']))
                # Add schedule information
                schedule_msg = ' | Schedule: {} (α={:.3f}, β={:.3f})'.format(
                    schedule['phase'], schedule['alpha'], schedule['beta'])
                log_msg += base_rl_msg + schedule_msg
        
        print(log_msg)
        
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
            # Add RL-related metrics only when RL is enabled and statistics exist
            if rl_enabled and 'valid_eval' in epoch_stats and 'policy_total_loss' in epoch_stats:
                rl_metrics = {
                    'valid_time': np.mean(epoch_stats['valid_time']),
                    'valid_eval': np.mean(epoch_stats['valid_eval']),
                    'valid_dist': np.mean(epoch_stats['valid_dist']),
                    'valid_ineq_max': np.mean(epoch_stats['valid_ineq_max']),
                    'valid_ineq_mean': np.mean(epoch_stats['valid_ineq_mean']),
                    'valid_ineq_num_viol_0': np.mean(epoch_stats['valid_ineq_num_viol_0']),
                    'valid_eq_max': np.mean(epoch_stats['valid_eq_max']),
                    'valid_eq_mean': np.mean(epoch_stats['valid_eq_mean']),
                    'policy_total_loss': np.mean(epoch_stats['policy_total_loss']),
                    'policy_bc_loss': np.mean(epoch_stats['policy_bc_loss']),
                    'policy_ql_loss': np.mean(epoch_stats['policy_ql_loss']),
                    'rl_improvement': np.mean(epoch_stats['valid_initial_eval']) - np.mean(epoch_stats['valid_eval'])
                }
                
                # Add schedule information
                # schedule = compute_bc_ql_schedule(i, args)
                schedule_phase_map = {
                    'Pure BC': 0,
                    'Final BC+QL': 2,
                    'Fixed': 3
                }
                # Handle transition phase (contains "Transition" in name)
                if 'Transition' in schedule['phase']:
                    schedule_phase_num = 1
                else:
                    schedule_phase_num = schedule_phase_map.get(schedule['phase'], 3)
                rl_metrics.update({
                    'schedule_alpha': schedule['alpha'],
                    'schedule_beta': schedule['beta'],
                    'schedule_phase': schedule_phase_num,
                    'schedule_progress': schedule['progress']
                })
                wandb_log_dict.update(rl_metrics)
            
            wandb.log(wandb_log_dict)

        if args['saveAllStats']:
            if i == 0:
                # Initialize stats with all keys from first epoch
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                # Handle keys that might be missing in some epochs
                for key in epoch_stats.keys():
                    if key in stats.keys():
                        # Key exists, concatenate
                        stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
                    else:
                        # Key is new, initialize with zeros for previous epochs
                        current_value = np.array(epoch_stats[key])
                        zeros_shape = (i,) + current_value.shape if current_value.ndim > 0 else (i,)
                        zeros = np.zeros(zeros_shape)
                        stats[key] = np.concatenate((zeros, np.expand_dims(current_value, axis=0)), axis=0)
        else:
            stats = epoch_stats

        if (i % args['resultsSaveFreq'] == 0):
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, 'initializer_net.dict'), 'wb') as f:
                torch.save(initializer_net.state_dict(), f)
            with open(os.path.join(save_dir, 'policy_net.dict'), 'wb') as f:
                torch.save(policy_net.state_dict(), f)

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'initializer_net.dict'), 'wb') as f:
        torch.save(initializer_net.state_dict(), f)
    with open(os.path.join(save_dir, 'policy_net.dict'), 'wb') as f:
        torch.save(policy_net.state_dict(), f)
    
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

    # Check if RL training has started and apply policy to improve solution
    if policy_net is not None and rl_enabled:
        # Apply policy to improve the solution
        Y_current = Y_initial.clone()
        for step in range(args['episodeLength']):
            with torch.no_grad():
                # Create state by concatenating X and Y_current
                state = torch.cat([X, Y_current], dim=1)
                # Sample action from policy
                # action, _ = policy_net.sample_action(state)
                action = policy_net.sample_action(state)
                # Apply action to update Y
                Y_current = Y_current + action
        Y = Y_current  # Use the improved Y from policy steps
        end_time = time.time()

        # Record final solution metrics (after RL if applicable)
        dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
        dict_agg(stats, make_prefix('steps'), args['episodeLength'])
        dict_agg(stats, make_prefix('loss'), loss_w_soft_penalty(data, X, Y, args).detach().cpu().numpy())
        if hasattr(data, 'data_all'):
            # This is a SimpleProblemMultiParam dataset
            dict_agg(stats, make_prefix('eval'), data.obj_fn(Y, X).detach().cpu().numpy())
        else:
            # This is a regular SimpleProblem dataset
            dict_agg(stats, make_prefix('eval'), data.obj_fn(Y).detach().cpu().numpy())
        dict_agg(stats, make_prefix('dist'), torch.norm(Y - Y_initial, dim=1).detach().cpu().numpy())  # Distance from initial to final
        dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Y), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_num_viol_0'),
                 torch.sum(data.ineq_dist(X, Y) > eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_num_viol_1'),
                 torch.sum(data.ineq_dist(X, Y) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_num_viol_2'),
                 torch.sum(data.ineq_dist(X, Y) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_max'),
                 torch.max(torch.abs(data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Y)), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_num_viol_0'),
                 torch.sum(torch.abs(data.eq_resid(X, Y)) > eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_num_viol_1'),
                 torch.sum(torch.abs(data.eq_resid(X, Y)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_num_viol_2'),
                 torch.sum(torch.abs(data.eq_resid(X, Y)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
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
            return self._data.complete_partial(x, out)
        else:
            return self._data.process_output(x, out)


if __name__=='__main__':
    main()
