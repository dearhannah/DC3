import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from typing import Tuple, Dict, List

# Set default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
print(f"Using device: {DEVICE}")

class LinearProgramDataset(Dataset):
    """Dataset for linear program data (A, b, x)"""
    def __init__(self, A_data: torch.Tensor, b_data: torch.Tensor, x_data: torch.Tensor):
        self.A_data = A_data  # Shape: (n_samples, dim_a, dim_a)
        self.b_data = b_data  # Shape: (n_samples, dim_b)
        self.x_data = x_data  # Shape: (n_samples, dim_b)
        
    def __len__(self):
        return len(self.A_data)
    
    def __getitem__(self, idx):
        # Flatten A for input to MLP
        A_flat = self.A_data[idx].flatten()  # Shape: (dim_a * dim_a,)
        return torch.cat([A_flat, self.b_data[idx]]), self.x_data[idx]

class MLPSolver(nn.Module):
    """MLP model to solve linear equations Ax = b"""
    def __init__(self, dim_a: int, dim_b: int, hidden_sizes: List[int] = None):
        super().__init__()
        
        # Adaptive hidden sizes based on problem dimension
        if hidden_sizes is None:
            base_size = max(256, dim_a * dim_a // 2)
            hidden_sizes = [base_size, base_size // 2, base_size // 4]
        
        # Input dimension: A (flattened) + b
        input_dim = dim_a * dim_a + dim_b
        output_dim = dim_b
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

def generate_linear_programs(n_samples: int, dim_a: int, dim_b: int, 
                           condition_number_range: Tuple[float, float] = (1.0, 100.0)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate n_samples linear programs Ax = b"""
    assert dim_a == dim_b, "For square systems, dim_a must equal dim_b"
    
    print(f"  Generating {n_samples} linear programs with A: {dim_a}x{dim_a}")
    
    A_data = torch.zeros(n_samples, dim_a, dim_a, dtype=torch.float32)
    b_data = torch.zeros(n_samples, dim_b, dtype=torch.float32)
    
    for i in range(n_samples):
        # Generate a well-conditioned matrix A
        Q, _ = torch.linalg.qr(torch.randn(dim_a, dim_a))
        
        # Generate diagonal matrix with controlled condition number
        min_cond, max_cond = condition_number_range
        cond_num = torch.exp(torch.rand(1) * torch.log(torch.tensor(max_cond / min_cond))) * min_cond
        
        # Create diagonal values with specified condition number
        d_max = 1.0
        d_min = d_max / cond_num
        diagonal = torch.exp(torch.rand(dim_a) * torch.log(torch.tensor(d_max / d_min))) * d_min
        D = torch.diag(diagonal)
        
        A = Q @ D @ Q.T
        
        # Generate random b vector
        b = torch.randn(dim_b)
        
        A_data[i] = A
        b_data[i] = b
    
    return A_data, b_data

def solve_linear_systems_batched(A_data: torch.Tensor, b_data: torch.Tensor, 
                                batch_size: int = 200) -> Tuple[torch.Tensor, float]:
    """Solve linear systems Ax = b in batches using torch.linalg.solve - speed focused"""
    n_samples = A_data.shape[0]
    dim_b = b_data.shape[1]
    
    x_solutions = torch.zeros(n_samples, dim_b, dtype=torch.float32)
    total_time = 0.0
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_A = A_data[i:end_idx].to(DEVICE)
        batch_b = b_data[i:end_idx].to(DEVICE)
        
        start_time = time.time()
        
        try:
            batch_x = torch.linalg.solve(batch_A, batch_b)
        except torch.linalg.LinAlgError:
            batch_x = torch.linalg.pinv(batch_A) @ batch_b.unsqueeze(-1)
            batch_x = batch_x.squeeze(-1)
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        # Store solutions
        x_solutions[i:end_idx] = batch_x.cpu()
    
    return x_solutions, total_time

def train_mlp_solver(A_data: torch.Tensor, b_data: torch.Tensor, x_data: torch.Tensor,
                    dim_a: int, dim_b: int, loss_threshold: float = 0.001,
                    batch_size: int = 32, max_epochs: int = 500,
                    learning_rate: float = 1e-3) -> MLPSolver:
    """Train MLP model to solve linear equations - simplified"""
    
    # Create dataset and dataloader
    dataset = LinearProgramDataset(A_data, b_data, x_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MLPSolver(dim_a, dim_b).to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # criterion = nn.MSELoss()
    
    # best_val_loss = float('inf')
    # patience_counter = 0
    # max_patience = 15
    
    # for epoch in range(max_epochs):
    #     # Training phase
    #     model.train()
    #     train_losses = []
        
    #     for batch_input, batch_target in train_loader:
    #         batch_input, batch_target = batch_input.to(DEVICE), batch_target.to(DEVICE)
            
    #         optimizer.zero_grad()
    #         predictions = model(batch_input)
    #         loss = criterion(predictions, batch_target)
    #         loss.backward()
    #         optimizer.step()
    #         train_losses.append(loss.item())
        
    #     avg_train_loss = np.mean(train_losses)
        
    #     # Validation phase
    #     model.eval()
    #     val_losses = []
        
    #     with torch.no_grad():
    #         for batch_input, batch_target in val_loader:
    #             batch_input, batch_target = batch_input.to(DEVICE), batch_target.to(DEVICE)
    #             predictions = model(batch_input)
    #             loss = criterion(predictions, batch_target)
    #             val_losses.append(loss.item())
        
    #     avg_val_loss = np.mean(val_losses)
        
    #     # Early stopping check
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         patience_counter = 0
    #         torch.save(model.state_dict(), f'best_mlp_solver_dim{dim_a}.pth')
    #     else:
    #         patience_counter += 1
        
    #     # Check if loss threshold is reached
    #     if avg_val_loss <= loss_threshold:
    #         break
        
    #     # Early stopping
    #     if patience_counter >= max_patience:
    #         break
    
    # # Load best model
    # model.load_state_dict(torch.load(f'best_mlp_solver_dim{dim_a}.pth'))
    
    return model

def predict_with_mlp_batched(model: MLPSolver, A_data: torch.Tensor, b_data: torch.Tensor,
                           batch_size: int = 200) -> Tuple[torch.Tensor, float]:
    """Use trained MLP model to predict solutions in batches - speed focused"""
    n_samples = A_data.shape[0]
    dim_b = b_data.shape[1]
    
    model.eval()
    predictions = torch.zeros(n_samples, dim_b, dtype=torch.float32)
    prediction_time = 0.0
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            
            # Prepare batch input
            batch_A = A_data[i:end_idx]
            batch_b = b_data[i:end_idx]
            
            # Flatten A and concatenate with b
            batch_A_flat = batch_A.flatten(start_dim=1)  # Shape: (batch_size, dim_a*dim_a)
            batch_input = torch.cat([batch_A_flat, batch_b], dim=1).to(DEVICE)
            
            start_time = time.time()
            batch_predictions = model(batch_input)
            end_time = time.time()
            
            prediction_time += (end_time - start_time)
            predictions[i:end_idx] = batch_predictions.cpu()
    
    return predictions, prediction_time

def benchmark_dimension(dim: int, n_samples: int = 5000, batch_size: int = 200, 
                       loss_threshold: float = 0.001) -> Dict:
    """Benchmark both methods for a specific dimension"""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING DIMENSION {dim}×{dim}")
    print(f"{'='*60}")
    
    # Generate datasets
    print(f"1. Generating datasets...")
    A_train, b_train = generate_linear_programs(n_samples, dim, dim)
    A_test, b_test = generate_linear_programs(n_samples, dim, dim)
    
    # Solve training data for MLP training
    print(f"2. Solving training data...")
    x_train_true, _ = solve_linear_systems_batched(A_train, b_train, batch_size)
    
    # Benchmark direct solver on test data
    print(f"3. Benchmarking direct solver...")
    x_test_true, direct_time = solve_linear_systems_batched(A_test, b_test, batch_size)
    
    # Train MLP
    print(f"4. Training MLP...")
    model = train_mlp_solver(A_train, b_train, x_train_true, dim, dim, 
                           loss_threshold=loss_threshold, max_epochs=300)
    
    # Benchmark MLP on test data
    print(f"5. Benchmarking MLP...")
    _, mlp_time = predict_with_mlp_batched(model, A_test, b_test, batch_size)
    
    # Calculate metrics
    direct_avg = direct_time / n_samples * 1000  # ms per problem
    mlp_avg = mlp_time / n_samples * 1000      # ms per problem
    speedup = direct_time / mlp_time
    
    results = {
        'dimension': dim,
        'n_samples': n_samples,
        'direct_total_time': direct_time,
        'mlp_total_time': mlp_time,
        'direct_avg_ms': direct_avg,
        'mlp_avg_ms': mlp_avg,
        'speedup': speedup
    }
    
    print(f"\nResults for {dim}×{dim}:")
    print(f"  Direct solver: {direct_time:.4f}s ({direct_avg:.4f}ms per problem)")
    print(f"  MLP prediction: {mlp_time:.4f}s ({mlp_avg:.4f}ms per problem)")
    print(f"  Speedup: {speedup:.2f}x")
    
    return results

def main():
    """Main function - multi-dimension speed benchmark"""
    
    # Configuration
    DIMENSIONS = [10, 30, 50, 70, 90]
    N_SAMPLES = 10000  # Reduced for faster benchmarking across multiple dimensions
    BATCH_SIZE = 200
    LOSS_THRESHOLD = 0.001
    
    print("="*80)
    print("MULTI-DIMENSION LINEAR EQUATION SOLVER SPEED BENCHMARK")
    print("="*80)
    print(f"Testing dimensions: {DIMENSIONS}")
    print(f"Samples per dimension: {N_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    
    all_results = []
    
    # Benchmark each dimension
    for dim in DIMENSIONS:
        try:
            results = benchmark_dimension(dim, N_SAMPLES, BATCH_SIZE, LOSS_THRESHOLD)
            all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking dimension {dim}: {e}")
            continue
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SPEED BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dim':<5} {'Direct(s)':<10} {'MLP(s)':<10} {'Direct(ms)':<12} {'MLP(ms)':<10} {'Speedup':<8}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['dimension']:<5} "
              f"{result['direct_total_time']:<10.4f} "
              f"{result['mlp_total_time']:<10.4f} "
              f"{result['direct_avg_ms']:<12.4f} "
              f"{result['mlp_avg_ms']:<10.4f} "
              f"{result['speedup']:<8.2f}x")
    
    # Analyze trends
    print(f"\n{'='*80}")
    print("TREND ANALYSIS")
    print(f"{'='*80}")
    
    if len(all_results) >= 2:
        print("Direct Solver Scaling:")
        for i in range(1, len(all_results)):
            prev_dim = all_results[i-1]['dimension']
            curr_dim = all_results[i]['dimension']
            prev_time = all_results[i-1]['direct_avg_ms']
            curr_time = all_results[i]['direct_avg_ms']
            scaling = curr_time / prev_time
            theoretical_scaling = (curr_dim / prev_dim) ** 3  # O(n^3) for matrix solve
            print(f"  {prev_dim}→{curr_dim}: {scaling:.2f}x slower (theoretical O(n³): {theoretical_scaling:.2f}x)")
        
        print("\nMLP Prediction Scaling:")
        for i in range(1, len(all_results)):
            prev_dim = all_results[i-1]['dimension']
            curr_dim = all_results[i]['dimension']
            prev_time = all_results[i-1]['mlp_avg_ms']
            curr_time = all_results[i]['mlp_avg_ms']
            scaling = curr_time / prev_time
            print(f"  {prev_dim}→{curr_dim}: {scaling:.2f}x slower")
        
        print("\nSpeedup Trend:")
        for result in all_results:
            status = "✓" if result['speedup'] > 1.0 else "✗"
            print(f"  {result['dimension']}×{result['dimension']}: {result['speedup']:.2f}x {status}")
        
        # Find optimal dimension range
        best_speedup = max(all_results, key=lambda x: x['speedup'])
        print(f"\nBest speedup: {best_speedup['speedup']:.2f}x at dimension {best_speedup['dimension']}×{best_speedup['dimension']}")
    
    return all_results

if __name__ == "__main__":
    results = main() 