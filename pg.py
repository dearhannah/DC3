# Import necessary libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Set plot style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 1. 数据加载函数
def load_data(file_path):
    """Load data"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Successfully loaded data: {file_path}")
        return data
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return None

def show_info(data):
    """显示数据基本信息"""
    if data is None:
        return
    print("=" * 50)
    print("�� 数据信息")
    print("=" * 50)
    print(f"轮次: {data['epoch_num']}")
    print(f"样本数: {data['y_init'].shape[0]:,}")
    print(f"Y维度: {data['y_init'].shape[1]}")
    print(f"轨迹步数: {data['trajectory_actions'].shape[1]}")
    print(f"动作维度: {data['trajectory_actions'].shape[2]}")
    print("=" * 50)

# 2. Statistical analysis functions
def analyze_stats(data):
    """Analyze statistical information"""
    if data is None:
        return
    # Calculate norms
    y_init_norms = np.linalg.norm(data['y_init'], axis=1)
    y_target_norms = np.linalg.norm(data['y_target'], axis=1)
    distance_norms = np.linalg.norm(data['y_init'] - data['y_target'], axis=1)
    
    # Calculate raw value statistics
    y_init_raw = data['y_init'].flatten()
    y_target_raw = data['y_target'].flatten()
    distance_raw = (data['y_init'] - data['y_target']).flatten()
    
    print("📊 统计摘要")
    print("=" * 50)
    
    # Y data statistics (norms)
    print("\n🎯 Y Data (Norms):")
    for name, values in [('Y_init', y_init_norms), ('Y_target', y_target_norms), ('Distance', distance_norms)]:
        print(f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, range=[{np.min(values):.4f}, {np.max(values):.4f}]")
    
    # Y data statistics (raw values)
    print("\n📈 Y Data (Raw Values):")
    for name, values in [('Y_init', y_init_raw), ('Y_target', y_target_raw), ('Distance', distance_raw)]:
        print(f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, range=[{np.min(values):.4f}, {np.max(values):.4f}]")
    
    # Trajectory action statistics
    print("\n🎮 Trajectory Actions:")
    n_steps = data['trajectory_actions'].shape[1]
    for step in range(n_steps):
        step_actions = data['trajectory_actions'][:, step, :]
        action_norms = np.linalg.norm(step_actions, axis=1)
        action_abs = np.abs(step_actions).flatten()
        action_raw = step_actions.flatten()  # 原始值
        print(f"Step {step}: action_norm(mean={np.mean(action_norms):.4f}, std={np.std(action_norms):.4f}, range=[{np.min(action_norms):.4f}, {np.max(action_norms):.4f}]), action_abs(mean={np.mean(action_abs):.4f}, std={np.std(action_abs):.4f}, range=[{np.min(action_abs):.4f}, {np.max(action_abs):.4f}]), action_raw(mean={np.mean(action_raw):.4f}, std={np.std(action_raw):.4f}, range=[{np.min(action_raw):.4f}, {np.max(action_raw):.4f}])")
    
    print("=" * 50)

# 3. Visualization functions
def plot_y_analysis(data):
    """Y data visualization"""
    if data is None:
        return
    
    distance_raw = data['y_target'] - data['y_init']  # Raw distance values
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Y Data Analysis', fontsize=16, fontweight='bold')
    
    # Y_init raw value distribution (all dimensions)
    y_init_flat = data['y_init'].flatten()
    axes[0, 0].hist(y_init_flat, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].set_xlabel('Y_init Raw Values')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Y_init Raw Value Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distance raw value distribution (mean across dimensions)
    distance_raw_mean = np.mean(distance_raw, axis=1)
    axes[0, 1].hist(distance_raw_mean, bins=50, alpha=0.7, edgecolor='black', color='gold')
    axes[0, 1].set_xlabel('Distance Raw Value Mean')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distance Raw Value Distribution (Mean)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distance absolute value distribution (mean across dimensions)
    distance_abs_mean = np.mean(np.abs(distance_raw), axis=1)
    axes[0, 2].hist(distance_abs_mean, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
    axes[0, 2].set_xlabel('Distance Absolute Value Mean')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distance Absolute Value Distribution (Mean)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Y_target raw value distribution (all dimensions)
    y_target_flat = data['y_target'].flatten()
    axes[1, 0].hist(y_target_flat, bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[1, 0].set_xlabel('Y_target Raw Values')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Y_target Raw Value Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distance raw value distribution (all dimensions)
    distance_raw_flat = distance_raw.flatten()
    axes[1, 1].hist(distance_raw_flat, bins=50, alpha=0.7, edgecolor='black', color='plum')
    axes[1, 1].set_xlabel('Distance Raw Values')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distance Raw Value Distribution (All Dimensions)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Distance absolute value distribution (all dimensions)
    distance_abs_flat = np.abs(distance_raw).flatten()
    axes[1, 2].hist(distance_abs_flat, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 2].set_xlabel('Distance Absolute Values')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Distance Absolute Value Distribution (All Dimensions)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_trajectory_analysis(data):
    """Trajectory action visualization"""
    if data is None:
        return
    
    n_steps = data['trajectory_actions'].shape[1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trajectory Action Analysis', fontsize=16, fontweight='bold')
    
    # Action norm violin plot
    action_norms_per_step = []
    for step in range(n_steps):
        step_actions = data['trajectory_actions'][:, step, :]
        action_norms = np.linalg.norm(step_actions, axis=1)
        action_norms_per_step.append(action_norms)
    
    violin_parts = axes[0, 0].violinplot(action_norms_per_step, positions=range(n_steps), showmeans=True)
    axes[0, 0].set_xlabel('Trajectory Steps')
    axes[0, 0].set_ylabel('Action Norm')
    axes[0, 0].set_title('Action Norm Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Action absolute value violin plot
    action_abs_per_step = []
    for step in range(n_steps):
        step_actions = data['trajectory_actions'][:, step, :]
        action_abs = np.abs(step_actions).flatten()
        action_abs_per_step.append(action_abs)
    
    violin_parts = axes[0, 1].violinplot(action_abs_per_step, positions=range(n_steps), showmeans=True)
    axes[0, 1].set_xlabel('Trajectory Steps')
    axes[0, 1].set_ylabel('Action Absolute Value')
    axes[0, 1].set_title('Action Absolute Value Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Action raw value violin plot
    action_raw_per_step = []
    for step in range(n_steps):
        step_actions = data['trajectory_actions'][:, step, :]
        action_raw = step_actions.flatten()
        action_raw_per_step.append(action_raw)
    
    # Create violin plot data
    violin_data = []
    violin_labels = []
    for step in range(n_steps):
        violin_data.append(action_raw_per_step[step])
        violin_labels.extend([f'Step {step}'] * len(action_raw_per_step[step]))
    
    # Draw violin plot
    violin_parts = axes[0, 2].violinplot(violin_data, positions=range(n_steps), showmeans=True)
    axes[0, 2].set_xlabel('Trajectory Steps')
    axes[0, 2].set_ylabel('Action Raw Values')
    axes[0, 2].set_title('Action Raw Value Distribution (Violin Plot)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Action norm trend
    action_norm_means = [np.mean(norms) for norms in action_norms_per_step]
    action_norm_stds = [np.std(norms) for norms in action_norms_per_step]
    
    axes[1, 0].errorbar(range(n_steps), action_norm_means, yerr=action_norm_stds, 
                       marker='o', capsize=5, capthick=2, linewidth=2)
    axes[1, 0].set_xlabel('Trajectory Steps')
    axes[1, 0].set_ylabel('Action Norm')
    axes[1, 0].set_title('Action Norm Trend')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Action absolute value trend
    action_abs_means = [np.mean(abs_vals) for abs_vals in action_abs_per_step]
    action_abs_stds = [np.std(abs_vals) for abs_vals in action_abs_per_step]
    
    axes[1, 1].errorbar(range(n_steps), action_abs_means, yerr=action_abs_stds, 
                       marker='s', capsize=5, capthick=2, linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Trajectory Steps')
    axes[1, 1].set_ylabel('Action Absolute Value')
    axes[1, 1].set_title('Action Absolute Value Trend')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Action raw value trend
    action_raw_means = [np.mean(raw_vals) for raw_vals in action_raw_per_step]
    action_raw_stds = [np.std(raw_vals) for raw_vals in action_raw_per_step]
    
    axes[1, 2].errorbar(range(n_steps), action_raw_means, yerr=action_raw_stds, 
                       marker='^', capsize=5, capthick=2, linewidth=2, color='purple')
    axes[1, 2].set_xlabel('Trajectory Steps')
    axes[1, 2].set_ylabel('Action Raw Values')
    axes[1, 2].set_title('Action Raw Value Trend')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_correlation(data):
    """Correlation analysis"""
    if data is None:
        return
    
    # Calculate metrics
    y_init_norms = np.linalg.norm(data['y_init'], axis=1)
    y_target_norms = np.linalg.norm(data['y_target'], axis=1)
    distance_norms = np.linalg.norm(data['y_init'] - data['y_target'], axis=1)
    
    # Calculate action norms
    n_steps = data['trajectory_actions'].shape[1]
    step_action_norms = []
    for step in range(n_steps):
        step_actions = data['trajectory_actions'][:, step, :]
        action_norms = np.linalg.norm(step_actions, axis=1)
        step_action_norms.append(action_norms)
    
    # Create correlation matrix
    corr_data = {
        'Y_init': y_init_norms,
        'Y_target': y_target_norms,
        'Distance': distance_norms
    }
    
    for step in range(n_steps):
        corr_data[f'Action_{step}'] = step_action_norms[step]
    
    corr_df = pd.DataFrame(corr_data)
    corr_matrix = corr_df.corr()
    
    # Draw heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Variable Correlation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 4. One-click analysis function
def run_analysis(file_path):
    """Run complete analysis"""
    print("🚀 Starting data analysis...")
    print("=" * 50)
    
    # Load data
    data = load_data(file_path)
    if data is None:
        return
    
    # Display information
    show_info(data)
    
    # Statistical analysis
    analyze_stats(data)
    
    # Visualization
    print("\nGenerating charts...")
    plot_y_analysis(data)
    plot_trajectory_analysis(data)
    # plot_correlation(data)
    
    print("\n✅ Analysis completed!")

# Usage example
if __name__ == "__main__":
    # Replace with your file path
    file_path = "results/your_experiment/method/hash_value/timestamp_regular/last_epoch_data.pkl"
    run_analysis(file_path)