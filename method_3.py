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

# Add wandb import
try:
    import wandb
    wandb.login(key='165a3c367eb0a61ffa11da3cc13ace2b93313835')
    WANDB_AVAILABLE = True
except ImportError:
    print("WARNING: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description='DC3')
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
    parser.add_argument('--saveAllStats', type=str_to_bool,
        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
        help='how frequently (in terms of number of epochs) to save stats to file')
    parser.add_argument('--useMultiParam', type=str_to_bool, default=True,
        help='whether to use multi-parameter dataset format')
    parser.add_argument('--useSanity', type=str_to_bool, default=False,
        help='whether to use sanity experiment dataset (same parameters, different X)')
    parser.add_argument('--useWandb', type=str_to_bool, default=False,
        help='whether to use wandb logging')
    parser.add_argument('--wandbProject', type=str, default='dc3-experiments',
        help='wandb project name')
    parser.add_argument('--wandbRunName', type=str, default=None,
        help='wandb run name (auto-generated if not provided)')
    
    # RL-specific arguments
    parser.add_argument('--rlLr', type=float, default=1e-4,
        help='learning rate for RL actor network')
    parser.add_argument('--episodeLength', type=int, default=5,
        help='number of steps per RL episode')
    parser.add_argument('--rlThreshold', type=float, default=1.0,
        help='threshold for constraint violations to start RL training')

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
                  run_name = f"sanity-rlascorr-{args['probType']}-{time_str}"
              else:
                  run_name = f"regular-rlascorr-{args['probType']}-{time_str}"
        
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
    solver_step = args['lr']
    nepochs = args['epochs']
    batch_size = args['batchSize']

    train_dataset = TensorDataset(data.trainX)
    # train_dataset = TensorDataset(data.validX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Initialize networks
    initializer_net = NNSolver(data, args)  # Network 1: Initializer (supervised)
    actor_net = ActorNetwork(data, args)    # Network 2: Actor (RL)
    
    initializer_net.to(DEVICE)
    actor_net.to(DEVICE)
    
    # Optimizers
    initializer_opt = optim.Adam(initializer_net.parameters(), lr=solver_step)
    actor_opt = optim.Adam(actor_net.parameters(), lr=args['rlLr'])

    stats = {}
    for i in range(nepochs):
        epoch_stats = {}

        # Get valid loss for initializer
        initializer_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_net(data, Xvalid, initializer_net, args, 'valid', epoch_stats, actor_net)

        # Get test loss for initializer
        initializer_net.eval()
        for Xtest in test_loader:
            Xtest = Xtest[0].to(DEVICE)
            eval_net(data, Xtest, initializer_net, args, 'test', epoch_stats, actor_net)

        # Training loop
        initializer_net.train()
        actor_net.train()
        
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
            
            # Step 1: Train initializer network (supervised learning)
            initializer_opt.zero_grad()
            Yhat_train = initializer_net(Xtrain)
            train_loss = total_loss(data, Xtrain, Yhat_train, args)
            train_loss.sum().backward()
            initializer_opt.step()
            
            # Step 2: Use initializer to get starting Y, then train actor with direct maximization
            with torch.no_grad():
                Y_init = initializer_net(Xtrain)  # Get initial Y from initializer
            
            # Only start RL training after initializer has good performance
            # Check if valid_ineq_num_viol_0 exists and is <= threshold
            if 'valid_ineq_num_viol_0' in epoch_stats and np.mean(epoch_stats['valid_ineq_num_viol_0']) <= args['rlThreshold']:
                # Log when RL training starts
                if 'direct_avg_improvement' not in epoch_stats or np.mean(epoch_stats['direct_avg_improvement']) == 0.0:
                    print(f"Starting RL training at epoch {i} (constraint violations: {np.mean(epoch_stats['valid_ineq_num_viol_0']):.3f} <= {args['rlThreshold']})")
                
                # Direct maximization episode
                Y_current = Y_init.clone()
                episode_improvements = []
                
                for step in range(args['episodeLength']):
                    # Get direct score difference
                    score_diff, Y_pred = actor_net.get_direct_score_difference(data, Xtrain, Y_current, args)
                    
                    # Update actor network to maximize score difference
                    actor_opt.zero_grad()
                    policy_loss = -score_diff.mean()  # Negative because we want to maximize
                    policy_loss.backward()
                    actor_opt.step()
                    
                    # Update current Y for next step
                    Y_current = Y_pred.detach()  # Use predicted Y as new current
                    episode_improvements.append(score_diff.mean().item())
                
                # Record RL statistics
                dict_agg(epoch_stats, 'direct_avg_improvement', episode_improvements)
                # dict_agg(epoch_stats, 'direct_final_improvement', episode_improvements[-1])
            else:
                # Skip RL training, record zeros for statistics
                dict_agg(epoch_stats, 'direct_avg_improvement', [0.0]*args['episodeLength'])
                # dict_agg(epoch_stats, 'direct_final_improvement', [0.0]*args['episodeLength'])
            
            train_time = time.time() - start_time
            
            # Record statistics
            dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')

        # Create log message
        log_msg = 'Epoch {}: train loss {:.4f}, eval {:.4f} (init: {:.4f}), dist {:.4f}, ineq max {:.4f} (init: {:.4f}), ineq num viol {:.4f} (init: {:.4f}), eq max {:.4f} (init: {:.4f}), steps {}, time {:.4f}, direct avg improvement {:.4f}'.format(
            i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']), np.mean(epoch_stats['valid_initial_eval']),
            np.mean(epoch_stats['valid_dist']), np.mean(epoch_stats['valid_ineq_max']), np.mean(epoch_stats['valid_initial_ineq_max']),
            np.mean(epoch_stats['valid_ineq_num_viol_0']), np.mean(epoch_stats['valid_initial_ineq_num_viol_0']),
            np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_initial_eq_max']),
            np.mean(epoch_stats['valid_steps']), np.mean(epoch_stats['valid_time']), np.mean(epoch_stats['direct_avg_improvement']))
        
        print(log_msg)
        
        # Log to wandb if available
        if args.get('useWandb', False) and WANDB_AVAILABLE:
            wandb.log({
                'epoch': i,
                'train_loss': np.mean(epoch_stats['train_loss']),
                'valid_eval': np.mean(epoch_stats['valid_eval']),
                'valid_initial_eval': np.mean(epoch_stats['valid_initial_eval']),
                'valid_dist': np.mean(epoch_stats['valid_dist']),
                'valid_ineq_max': np.mean(epoch_stats['valid_ineq_max']),
                'valid_initial_ineq_max': np.mean(epoch_stats['valid_initial_ineq_max']),
                'valid_ineq_mean': np.mean(epoch_stats['valid_ineq_mean']),
                'valid_ineq_num_viol_0': np.mean(epoch_stats['valid_ineq_num_viol_0']),
                'valid_initial_ineq_num_viol_0': np.mean(epoch_stats['valid_initial_ineq_num_viol_0']),
                'valid_eq_max': np.mean(epoch_stats['valid_eq_max']),
                'valid_initial_eq_max': np.mean(epoch_stats['valid_initial_eq_max']),
                'valid_steps': np.mean(epoch_stats['valid_steps']),
                'valid_time': np.mean(epoch_stats['valid_time']),
                'train_time': np.mean(epoch_stats['train_time']),
                'direct_avg_improvement': np.mean(epoch_stats['direct_avg_improvement']),
                'direct_total_improvement': np.sum(epoch_stats['direct_avg_improvement'])
            })

        if args['saveAllStats']:
            if i == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            stats = epoch_stats

        if (i % args['resultsSaveFreq'] == 0):
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, 'initializer_net.dict'), 'wb') as f:
                torch.save(initializer_net.state_dict(), f)
            with open(os.path.join(save_dir, 'actor_net.dict'), 'wb') as f:
                torch.save(actor_net.state_dict(), f)

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'initializer_net.dict'), 'wb') as f:
        torch.save(initializer_net.state_dict(), f)
    with open(os.path.join(save_dir, 'actor_net.dict'), 'wb') as f:
        torch.save(actor_net.state_dict(), f)
    
    # Finish wandb run
    if args.get('useWandb', False) and WANDB_AVAILABLE:
        wandb.finish()
        print("Wandb run finished")
    
    return initializer_net, actor_net, stats

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
def eval_net(data, X, solver_net, args, prefix, stats, actor_net=None):
    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y_initial = solver_net(X)

    # Record initial solution metrics
    dict_agg(stats, make_prefix('initial_loss'), total_loss(data, X, Y_initial, args).detach().cpu().numpy())
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
    dict_agg(stats, make_prefix('initial_eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Y_initial)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Y_initial)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('initial_eq_num_viol_0'),
             torch.sum(torch.abs(data.eq_resid(X, Y_initial)) > eps_converge, dim=1).detach().cpu().numpy())

    # Check if RL training has started and apply RL steps if available
    Y = Y_initial  # Start with initial solution
    # if 'direct_avg_improvement' in stats and np.mean(stats['direct_avg_improvement']) > 0.0:
    if True:
        # Apply RL steps to improve the solution
        Y_current = Y_initial.clone()
        for step in range(args['episodeLength']):
            with torch.no_grad():
                score_diff, Y_pred = actor_net.get_direct_score_difference(data, X, Y_current, args)
                Y_current = Y_pred  # Update current Y for next step
        Y = Y_current  # Use the improved Y from RL steps
    end_time = time.time()

    # Record final solution metrics (after RL if applicable)
    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('steps'), args['episodeLength'])  # No correction steps
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Y, args).detach().cpu().numpy())
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

def total_loss(data, X, Y, args):
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

class ActorNetwork(nn.Module):
    """Actor network for RL that takes [X, Y] as input and outputs Y update"""
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        # Input dimension is X + Y concatenated
        input_dim = data.xdim + data.ydim
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

    def forward(self, x, y):
        # Concatenate X and Y for state representation
        state = torch.cat([x, y], dim=1)
        out = self.net(state)
 
        if self._args['useCompl']:
            if 'acopf' in self._args['probType']:
                out = nn.Sigmoid()(out)   # used to interpolate between max and min values
            return self._data.complete_partial(x, out)
        else:
            return self._data.process_output(x, out)

    def get_direct_score_difference(self, data, x, y_current, args):
        """Direct score difference maximization"""
        y_pred = self.forward(x, y_current)
        # Calculate score difference directly
        score_current = -total_loss(data, x, y_current, args)
        score_pred = -total_loss(data, x, y_pred, args)
        score_diff = score_pred - score_current
        return score_diff, y_pred

if __name__=='__main__':
    main()
