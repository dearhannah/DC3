#!/usr/bin/env python3

import torch
import pickle
import numpy as np
import sys
import os

# Add the current directory to Python path to import method_3
sys.path.append('.')

from method_3 import NNSolver, ActorNetwork, eval_net

def load_model_and_eval():
    # Model directory
    model_dir = "/sciclone/home/hmeng/research/DC3/results/SimpleProblemMultiParam-100-50-50-10000/method/c6143f84c352df20746b8566a58999869b1a5bae/2025Jul28-081801_sanity/"
    
    print(f"Loading models from: {model_dir}")
    
    # Load args
    with open(os.path.join(model_dir, 'args.dict'), 'rb') as f:
        args = pickle.load(f)
    
    print("Loaded args:", args)
    
    # Load data
    print("Loading data...")
    # Override the dataset path to use the specific sanity dataset
    args['dataPath'] = "/sciclone/home/hmeng/research/DC3/datasets/simple/random_simple_multi_param_sanity_dataset_var100_ineq50_eq50_ex10000"
    with open(args['dataPath'], 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded: {data.xdim} input dim, {data.ydim} output dim, {data.nknowns} knowns")
    
    # Initialize networks
    print("Initializing networks...")
    initializer_net = NNSolver(data,args)
    actor_net = ActorNetwork(data, args)
    
    # Load trained weights
    print("Loading trained weights...")
    initializer_net.load_state_dict(torch.load(os.path.join(model_dir, 'initializer_net.dict')))
    actor_net.load_state_dict(torch.load(os.path.join(model_dir, 'actor_net.dict')))
    
    # Set to evaluation mode
    initializer_net.eval()
    actor_net.eval()
    
    print("Models loaded successfully!")
    
    # Run evaluation
    print("\n" + "="*50)
    print("RUNNING EVALUATION")
    print("="*50)
    
    # Create stats dictionary for evaluation
    stats = {}
    
    # Run evaluation on test set
    eval_net(data, data.validX, initializer_net, args, 'valid', stats, actor_net)
    
    # Print key results
    print("\n" + "="*50)
    print("EVALUATION RESULTS (Validation Set)")
    print("="*50)
    
    # Print initial metrics (before RL)
    if 'valid_initial_loss' in stats:
        print(f"Initial Loss: {np.mean(stats['valid_initial_loss']):.6f}")
    if 'valid_initial_ineq_mean' in stats:
        print(f"Initial Inequality Violations (mean): {np.mean(stats['valid_initial_ineq_mean']):.6f}")
    if 'valid_initial_ineq_num_viol_0' in stats:
        print(f"Initial Inequality Violations (num): {np.mean(stats['valid_initial_ineq_num_viol_0']):.6f}")
    
    # Print final metrics (after RL)
    if 'valid_loss' in stats:
        print(f"Final Loss: {np.mean(stats['valid_loss']):.6f}")
    if 'valid_ineq_mean' in stats:
        print(f"Final Inequality Violations (mean): {np.mean(stats['valid_ineq_mean']):.6f}")
    if 'valid_ineq_num_viol_0' in stats:
        print(f"Final Inequality Violations (num): {np.mean(stats['valid_ineq_num_viol_0']):.6f}")
    
    # Print improvement metrics
    if 'valid_direct_avg_improvement' in stats:
        print(f"Average Improvement per RL Step: {np.mean(stats['valid_direct_avg_improvement']):.6f}")
    if 'valid_direct_final_improvement' in stats:
        print(f"Total Improvement: {np.mean(stats['valid_direct_final_improvement']):.6f}")
    if 'valid_dist' in stats:
        print(f"Average Distance from Initial Solution: {np.mean(stats['valid_dist']):.6f}")
    
    print("="*50)
    
    return stats

if __name__ == "__main__":
    stats = load_model_and_eval() 