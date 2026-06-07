import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Central configuration for the preprocessing pipeline"""


    TRAIN_PATH = "UNSW_NB15_training-set.csv"
    TEST_PATH = "UNSW_NB15_testing-set.csv"


    OUTPUT_DIR = Path("preprocessing_outputs")
    EDA_DIR = OUTPUT_DIR / "eda"
    MODELS_DIR = OUTPUT_DIR / "models"
    LOGS_DIR = OUTPUT_DIR / "logs"


    LABEL_COL = "label"
    ATTACK_CAT_COL = "attack_cat"


    MISSING_THRESHOLD = 0.40
    VARIANCE_THRESHOLD = 0.001

    RANDOM_STATE = 42

    def __init__(self):
        """Create output directories"""
        for d in [self.EDA_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

config = Config()


def load_data():


    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    print(f"\nTraining set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")

    with open(config.LOGS_DIR / "data_shapes.txt", "w") as f:
        f.write(f"Training set: {train_df.shape}\n")
        f.write(f"Testing set: {test_df.shape}\n")

    return train_df, test_df


def initial_eda(df, dataset_name="train"):

    print(f"\nShape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes.value_counts()}")

    unique_counts = df.nunique().sort_values()
    print("\nUnique values per column:")
    print(unique_counts)
    unique_counts.to_csv(config.EDA_DIR / f"{dataset_name}_unique_counts.csv")


    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    }).sort_values('missing_pct', ascending=False)

    print(f"\nMissing values:\n{missing_df[missing_df['missing_count'] > 0]}")
    missing_df.to_csv(config.EDA_DIR / f"{dataset_name}_missing_values.csv")

    if missing.sum() > 0:
        cols_with_missing = missing[missing > 0].index.tolist()
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[cols_with_missing].isnull(), cbar=True, yticklabels=False)
        plt.title(f"Missingness Heatmap - {dataset_name}")
        plt.tight_layout()
        plt.savefig(config.EDA_DIR / f"{dataset_name}_missingness_heatmap.png", dpi=150)
        plt.close()
        print("✓ Saved missingness heatmap")


    if config.ATTACK_CAT_COL in df.columns:
        plt.figure(figsize=(12, 6))
        attack_counts = df[config.ATTACK_CAT_COL].value_counts()
        attack_counts.plot(kind='bar')
        plt.title(f"Attack Category Distribution - {dataset_name}")
        plt.xlabel("Attack Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(config.EDA_DIR / f"{dataset_name}_attack_distribution.png", dpi=150)
        plt.close()
        print("\nAttack category distribution:")
        print(attack_counts)
        attack_counts.to_csv(config.EDA_DIR / f"{dataset_name}_attack_counts.csv")
        print("✓ Saved attack distribution plot")


    if config.LABEL_COL in df.columns:
        label_counts = df[config.LABEL_COL].value_counts()
        print("\nBinary label distribution:")
        print(label_counts)
        print(f"Attack ratio: {label_counts.get(1, 0) / len(df):.2%}")


    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [config.LABEL_COL, config.ATTACK_CAT_COL]]

    if numeric_cols:
        print(f"\nAnalyzing {len(numeric_cols)} numeric features...")
        percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        stats_list = []

        for col in numeric_cols:
            col_stats = {
                'feature': col,
                'dtype': str(df[col].dtype),
                'count': df[col].count(),
                'missing': df[col].isnull().sum(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'variance': df[col].var(),
            }
            for p in percentiles:
                col_stats[f'p{p}'] = df[col].quantile(p / 100)
            stats_list.append(col_stats)

        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(config.EDA_DIR / f"{dataset_name}_numeric_stats.csv", index=False)
        print("✓ Saved numeric statistics")

        n_features_to_plot = min(20, len(numeric_cols))
        fig, axes = plt.subplots(5, 4, figsize=(20, 15))
        axes = axes.ravel()
        for idx, col in enumerate(numeric_cols[:n_features_to_plot]):
            axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(col, fontsize=10)
            axes[idx].tick_params(labelsize=8)
        for idx in range(n_features_to_plot, len(axes)):
            axes[idx].axis('off')
        plt.suptitle(f"Feature Distributions (Sample) - {dataset_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(config.EDA_DIR / f"{dataset_name}_distributions_sample.png", dpi=150)
        plt.close()
        print("✓ Saved distribution plots")


    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != config.ATTACK_CAT_COL]

    if categorical_cols:
        print(f"\nAnalyzing {len(categorical_cols)} categorical features...")
        cat_stats = []
        for col in categorical_cols:
            cat_stats.append({
                'feature': col,
                'n_unique': df[col].nunique(),
                'n_missing': df[col].isnull().sum(),
                'most_common': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                'most_common_count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
            })
        cat_stats_df = pd.DataFrame(cat_stats)
        cat_stats_df.to_csv(config.EDA_DIR / f"{dataset_name}_categorical_stats.csv", index=False)
        print("✓ Saved categorical statistics")


    summary = {
        'dataset': dataset_name,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_numeric': len(numeric_cols),
        'n_categorical': len(categorical_cols),
        'total_missing': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


    summary_serializable = {
        k: (int(v) if isinstance(v, np.integer)
            else float(v) if isinstance(v, np.floating)
            else v)
        for k, v in summary.items()
    }

    with open(config.EDA_DIR / f"{dataset_name}_summary.json", "w") as f:
        json.dump(summary_serializable, f, indent=2)


def main():


    train_df, test_df = load_data()
    train_summary = initial_eda(train_df, "train")
    test_summary = initial_eda(test_df, "test")


    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"\nColumns in train but not in test: {set(train_df.columns) - set(test_df.columns)}")
    print(f"Columns in test but not in train: {set(test_df.columns) - set(train_df.columns)}")

    print("\n" + "="*80)
    print("PART 1 COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review EDA outputs in: {config.EDA_DIR}")
    print(f"2. Identify features to drop (identifiers, high missing, constants)")
    print(f"3. Run part2_feature_drop.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
