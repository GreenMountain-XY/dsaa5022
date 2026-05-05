import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('/Users/hangyu/.openclaw/workspace/dsaa5022/data/ethereum_fraud.csv')

# Feature engineering (same as in report)
def engineer_features(df):
    df = df.copy()
    eps = 1e-6
    
    df['sent_received_ratio'] = df['Sent tnx'] / (df['Received Tnx'] + eps)
    df['avg_transaction_value'] = df['total Ether sent'] / (df['Sent tnx'] + eps)
    df['contract_ratio'] = df['Number of Created Contracts'] / \
        (df['total transactions (including tnx to create contract'] + eps)
    df['balance_per_transaction'] = df['total ether balance'] / \
        (df['total transactions (including tnx to create contract'] + eps)
    
    df['total_time_active'] = df['Time Diff between first and last (Mins)']
    df['avg_sent_interval'] = df['Avg min between sent tnx']
    df['avg_received_interval'] = df['Avg min between received tnx']
    
    df['ether_flow'] = df['total ether received'] - df['total Ether sent']
    df['max_received_ratio'] = df['max value received '] / \
        (df['avg val received'] + eps)
    
    df['erc20_activity'] = df[' Total ERC20 tnxs'] / \
        (df['total transactions (including tnx to create contract'] + eps)
    
    df = df.fillna(0)
    return df

df = engineer_features(df)

# Define features to plot
derived_features = [
    'sent_received_ratio',
    'avg_transaction_value', 
    'contract_ratio',
    'balance_per_transaction',
    'ether_flow',
    'max_received_ratio',
    'erc20_activity'
]

# Custom x-axis limits for better visualization
# Format: (lower_percentile, upper_percentile) for clipping
clip_ranges = {
    'sent_received_ratio': (0, 95),      # Most data in 0-50 range
    'avg_transaction_value': (0, 95),    # Heavy right tail
    'contract_ratio': (0, 99),           # Most are near 0
    'balance_per_transaction': (1, 99),  # Has negative values
    'ether_flow': (1, 99),               # Has negative values, heavy tails
    'max_received_ratio': (0, 95),       # Heavy right tail
    'erc20_activity': (0, 95),           # Extreme outliers
}

# Split by class
df_normal = df[df['FLAG'] == 0]
df_fraud = df[df['FLAG'] == 1]

# Create figure
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Derived Features: Distribution Comparison with Mann-Whitney U Tests', 
             fontsize=18, fontweight='bold', y=0.98)

axes = axes.flatten()

for idx, feature in enumerate(derived_features):
    ax = axes[idx]
    
    # Get data
    normal_data = df_normal[feature].values
    fraud_data = df_fraud[feature].values
    
    # Smart clipping based on percentiles
    low_p, high_p = clip_ranges.get(feature, (0, 95))
    combined = np.concatenate([normal_data, fraud_data])
    
    # Calculate clip bounds
    lower = np.percentile(combined, low_p)
    upper = np.percentile(combined, high_p)
    
    # Add small padding
    padding = (upper - lower) * 0.05
    x_min = lower - padding
    x_max = upper + padding
    
    # Clip data for visualization
    normal_clip = np.clip(normal_data, x_min, x_max)
    fraud_clip = np.clip(fraud_data, x_min, x_max)
    
    # Plot KDE with better bandwidth
    bw_adjust = 0.5  # Smoother curves
    
    if len(set(normal_clip)) > 1:
        sns.kdeplot(data=normal_clip, ax=ax, color='#3498db', fill=True, 
                   alpha=0.25, label='Normal', warn_singular=False,
                   bw_adjust=bw_adjust, cut=0)
    else:
        ax.axvline(normal_clip[0], color='#3498db', linewidth=3, label='Normal', alpha=0.7)
        
    if len(set(fraud_clip)) > 1:
        sns.kdeplot(data=fraud_clip, ax=ax, color='#e74c3c', fill=True, 
                   alpha=0.25, label='Fraud', warn_singular=False,
                   bw_adjust=bw_adjust, cut=0)
    else:
        ax.axvline(fraud_clip[0], color='#e74c3c', linewidth=3, label='Fraud', alpha=0.7)
    
    # Set x-axis limits
    ax.set_xlim(x_min, x_max)
    
    # Mann-Whitney U test on original (unclipped) data
    try:
        statistic, p_value = mannwhitneyu(normal_data, fraud_data, alternative='two-sided')
        if p_value < 1e-300:
            p_str = "p < 1×10⁻³⁰⁰"
        elif p_value < 0.001:
            p_str = f"p = {p_value:.2e}"
        else:
            p_str = f"p = {p_value:.4f}"
    except:
        p_str = "p = N/A"
    
    ax.set_title(f'{feature.replace("_", " ").title()}\n{p_str}', 
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis to avoid scientific notation where possible
    ax.ticklabel_format(style='plain', axis='x')

# Hide the last (8th) subplot
axes[7].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/hangyu/.openclaw/workspace/dsaa5022/figures/v2_fig2_derived_kde_significance.png', 
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("Figure saved successfully with improved visualization!")
