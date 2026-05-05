import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/hangyu/.openclaw/workspace/dsaa5022/data/ethereum_fraud.csv')

# Feature engineering (same as in report)
def engineer_features(df):
    df = df.copy()
    eps = 1e-6
    
    # 比率特征
    df['sent_received_ratio'] = df['Sent tnx'] / (df['Received Tnx'] + eps)
    df['avg_transaction_value'] = df['total Ether sent'] / (df['Sent tnx'] + eps)
    df['contract_ratio'] = df['Number of Created Contracts'] / \
        (df['total transactions (including tnx to create contract'] + eps)
    df['balance_per_transaction'] = df['total ether balance'] / \
        (df['total transactions (including tnx to create contract'] + eps)
    
    # 时间特征
    df['total_time_active'] = df['Time Diff between first and last (Mins)']
    df['avg_sent_interval'] = df['Avg min between sent tnx']
    df['avg_received_interval'] = df['Avg min between received tnx']
    
    # ETH流动
    df['ether_flow'] = df['total ether received'] - df['total Ether sent']
    df['max_received_ratio'] = df['max value received '] / \
        (df['avg val received'] + eps)
    
    # ERC20
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

# Split by class
df_normal = df[df['FLAG'] == 0]
df_fraud = df[df['FLAG'] == 1]

# Create figure
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Derived Features: Distribution Comparison with Statistical Tests', 
             fontsize=16, fontweight='bold', y=0.98)

axes = axes.flatten()

for idx, feature in enumerate(derived_features):
    ax = axes[idx]
    
    # Get data
    normal_data = df_normal[feature].values
    fraud_data = df_fraud[feature].values
    
    # Remove extreme outliers for visualization (clip to 99.9th percentile)
    combined = np.concatenate([normal_data, fraud_data])
    p99 = np.percentile(combined, 99.9)
    p01 = np.percentile(combined, 0.1)
    
    normal_clip = np.clip(normal_data, p01, p99)
    fraud_clip = np.clip(fraud_data, p01, p99)
    
    # Plot KDE
    if len(set(normal_clip)) > 1 and len(set(fraud_clip)) > 1:
        sns.kdeplot(data=normal_clip, ax=ax, color='#3498db', fill=True, alpha=0.3, label='Normal', warn_singular=False)
        sns.kdeplot(data=fraud_clip, ax=ax, color='#e74c3c', fill=True, alpha=0.3, label='Fraud', warn_singular=False)
    else:
        # If all values are the same, plot a vertical line
        if len(set(normal_clip)) == 1:
            ax.axvline(normal_clip[0], color='#3498db', linewidth=2, label='Normal')
        if len(set(fraud_clip)) == 1:
            ax.axvline(fraud_clip[0], color='#e74c3c', linewidth=2, label='Fraud')
    
    # Mann-Whitney U test
    try:
        statistic, p_value = mannwhitneyu(normal_data, fraud_data, alternative='two-sided')
        p_str = f"p={p_value:.2e}" if p_value > 1e-300 else f"p < 1e-300"
    except:
        p_str = "p=N/A"
    
    ax.set_title(f'{feature}\n{p_str}', fontsize=11, fontweight='bold')
    ax.set_xlabel(feature, fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide the last (8th) subplot
axes[7].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/Users/hangyu/.openclaw/workspace/dsaa5022/figures/v2_fig2_derived_kde_significance.png', 
            dpi=150, bbox_inches='tight')
plt.close()

print("Figure saved successfully!")

# Also print some stats for erc20_activity to diagnose
print("\n=== erc20_activity stats ===")
print(f"Normal - mean: {df_normal['erc20_activity'].mean():.4f}, std: {df_normal['erc20_activity'].std():.4f}")
print(f"Normal - min: {df_normal['erc20_activity'].min():.4f}, max: {df_normal['erc20_activity'].max():.4f}")
print(f"Normal - unique values: {len(set(df_normal['erc20_activity']))}")
print(f"Fraud - mean: {df_fraud['erc20_activity'].mean():.4f}, std: {df_fraud['erc20_activity'].std():.4f}")
print(f"Fraud - min: {df_fraud['erc20_activity'].min():.4f}, max: {df_fraud['erc20_activity'].max():.4f}")
print(f"Fraud - unique values: {len(set(df_fraud['erc20_activity']))}")
