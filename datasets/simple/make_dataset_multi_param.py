import numpy as np
import pickle
import torch
import sys
import os
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import SimpleProblemMultiParam, SimpleProblem

torch.set_default_dtype(torch.float64)

num_var = 100
num_ineq = 50
num_eq = 50
num_examples = 10000

valid_frac = 0.0833
test_frac = 0.0833

general_partial_vars = np.arange(num_var - num_eq)
general_other_vars = np.arange(num_var - num_eq, num_var)

# SANITY EXPERIMENT: Set to True for sanity experiment (same parameters, different X)
SANITY_MODE = True

if SANITY_MODE:
    print("SANITY MODE: Using same parameters for all examples")
    # Generate ONE set of parameters
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)

    # Find valid found_partial_vars/found_other_vars
    det = 0
    tries = 0
    while abs(det) < 0.0001 and tries < 100:
        found_partial_vars = np.random.choice(num_var, num_var - num_eq, replace=False)
        found_other_vars = np.setdiff1d(np.arange(num_var), found_partial_vars)
        det = np.linalg.det(A[:, found_other_vars])
        tries += 1
    if tries == 100:
        raise Exception('Could not find invertible A_other')

    # Compute permutation to map found_partial_vars -> general_partial_vars, found_other_vars -> general_other_vars
    perm = np.empty(num_var, dtype=int)
    perm[general_partial_vars] = found_partial_vars
    perm[general_other_vars] = found_other_vars

    # Permute all parameters
    Q = Q[perm][:, perm]
    p = p[perm]
    A = A[:, perm]
    G = G[:, perm]

Qs = np.zeros((num_examples, num_var))
ps = np.zeros((num_examples, num_var))
As = np.zeros((num_examples, num_eq, num_var))
Gs = np.zeros((num_examples, num_ineq, num_var))
hs = np.zeros((num_examples, num_ineq))
Xs = np.zeros((num_examples, num_eq))
Ys = np.zeros((num_examples, num_var))

for i in tqdm(range(num_examples)):
    if SANITY_MODE:
        # Use the SAME parameters for all examples
        Qs[i] = np.diag(Q)
        ps[i] = p
        As[i] = A
        Gs[i] = G
        hs[i] = h
        X = np.random.uniform(-1, 1, size=(num_eq,))
        Xs[i] = X
    else:
        # Generate DIFFERENT parameters for each example (original behavior)
        Q = np.diag(np.random.random(num_var))
        p = np.random.random(num_var)
        A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
        X = np.random.uniform(-1, 1, size=(num_eq,))
        G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
        h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)

        # Find valid found_partial_vars/found_other_vars
        det = 0
        tries = 0
        while abs(det) < 0.0001 and tries < 100:
            found_partial_vars = np.random.choice(num_var, num_var - num_eq, replace=False)
            found_other_vars = np.setdiff1d(np.arange(num_var), found_partial_vars)
            det = np.linalg.det(A[:, found_other_vars])
            tries += 1
        if tries == 100:
            raise Exception('Could not find invertible A_other')

        # Compute permutation to map found_partial_vars -> general_partial_vars, found_other_vars -> general_other_vars
        perm = np.empty(num_var, dtype=int)
        perm[general_partial_vars] = found_partial_vars
        perm[general_other_vars] = found_other_vars

        # Permute all parameters
        Q = Q[perm][:, perm]
        p = p[perm]
        A = A[:, perm]
        G = G[:, perm]
        # X, h do not change

        Qs[i] = np.diag(Q)
        ps[i] = p
        As[i] = A
        Gs[i] = G
        hs[i] = h
        Xs[i] = X

    problem = SimpleProblem(Q, p, A, G, h, X[None, :])
    Y = problem.opt_solve(problem.X)[0][0]

    Ys[i] = Y

# Build dims dict
dims = {
    'Q': (num_var,),  # Q is now just diagonal elements
    'p': (ps.shape[1],),
    'A': (As.shape[1], As.shape[2]),
    'G': (Gs.shape[1], Gs.shape[2]),
    'h': (hs.shape[1],),
    'X': (Xs.shape[1],)
}

# Build data_all (concatenation of Q_diag, p, A, G, h, X)
# Extract only diagonal elements of Q since Q is diagonal
# Q_diags = np.array([np.diag(Q) for Q in Qs])

data_all = np.concatenate([
    Qs.reshape(num_examples, -1),  # Only diagonal elements of Q
    ps.reshape(num_examples, -1),
    As.reshape(num_examples, -1),
    Gs.reshape(num_examples, -1),
    hs.reshape(num_examples, -1),
    Xs.reshape(num_examples, -1)
], axis=1)

problem_multi = SimpleProblemMultiParam(
    data_all, Ys, dims,
    partial_vars=general_partial_vars,
    other_vars=general_other_vars,
    valid_frac=valid_frac,
    test_frac=test_frac
)

# Choose filename based on mode
if SANITY_MODE:
    filename = f"random_simple_multi_param_sanity_dataset_var{num_var}_ineq{num_ineq}_eq{num_eq}_ex{num_examples}"
else:
    filename = f"random_simple_multi_param_dataset_var{num_var}_ineq{num_ineq}_eq{num_eq}_ex{num_examples}"

with open(filename, 'wb') as f:
    pickle.dump(problem_multi, f) 