import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class Config:
    OUTPUT_DIR = Path("preprocessing_outputs")
    MODELS_DIR = OUTPUT_DIR / "models"
    LOGS_DIR = OUTPUT_DIR / "logs"

    LABEL_COL = "label"
    ATTACK_CAT_COL = "attack_cat"

    VAL_SIZE = 0.18

    RANDOM_STATE = 42

    def __init__(self):
        """Create output directories"""
        for d in [self.OUTPUT_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

config = Config()


def load_cleaned_data():
    """Load the cleaned datasets from Part 2"""
    print("Loading cleaned datasets...")

    train_df = pd.read_csv(config.OUTPUT_DIR / "train_cleaned.csv")
    test_df = pd.read_csv(config.OUTPUT_DIR / "test_cleaned.csv")

    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"\nNote: Test set will remain completely separate and untouched!")

    return train_df, test_df

def check_class_distribution(df, name="dataset"):
    """Display class distribution for stratification validation"""
    print(f"\n{name} class distribution:")

    if config.LABEL_COL in df.columns:
        label_counts = df[config.LABEL_COL].value_counts()
        label_dist = df[config.LABEL_COL].value_counts(normalize=True)
        print(f"\nBinary labels (0=Normal, 1=Attack):")
        for label, count in label_counts.items():
            pct = label_dist[label] * 100
            print(f"  {label}: {count:,} ({pct:.2f}%)")

    if config.ATTACK_CAT_COL in df.columns:
        attack_counts = df[config.ATTACK_CAT_COL].value_counts()
        attack_dist = df[config.ATTACK_CAT_COL].value_counts(normalize=True)
        print(f"\nAttack categories (all):")
        for cat, count in attack_counts.items():
            pct = attack_dist[cat] * 100
            print(f"  {cat}: {count:,} ({pct:.2f}%)")

def main():
    print("\n" + "=" * 80)
    print("UNSW-NB15 PREPROCESSING - PART 3: TRAIN/VALIDATION SPLIT")
    print("=" * 80)

    train_df_cleaned, test_df = load_cleaned_data()

    if config.LABEL_COL not in train_df_cleaned.columns:
        raise ValueError(f"Target column '{config.LABEL_COL}' not found in train_cleaned.csv")

    drop_cols = [config.LABEL_COL]
    if config.ATTACK_CAT_COL in train_df_cleaned.columns:
        drop_cols.append(config.ATTACK_CAT_COL)

    X_train, X_val, y_train, y_val = train_test_split(
        train_df_cleaned.drop(columns=drop_cols),
        train_df_cleaned[config.LABEL_COL],
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=train_df_cleaned[config.LABEL_COL]
    )


    train_split_df = pd.concat([X_train, y_train], axis=1)
    if config.ATTACK_CAT_COL in train_df_cleaned.columns:
        train_split_df[config.ATTACK_CAT_COL] = train_df_cleaned.loc[X_train.index, config.ATTACK_CAT_COL]

    val_split_df = pd.concat([X_val, y_val], axis=1)
    if config.ATTACK_CAT_COL in train_df_cleaned.columns:
        val_split_df[config.ATTACK_CAT_COL] = train_df_cleaned.loc[X_val.index, config.ATTACK_CAT_COL]

    print(f"\nOriginal cleaned training shape: {train_df_cleaned.shape}")
    print(f"New training split shape: {train_split_df.shape}")
    print(f"Validation split shape: {val_split_df.shape}")
    print(f"Original test shape (unchanged): {test_df.shape}")


    check_class_distribution(train_df_cleaned, "Original Cleaned Train")
    check_class_distribution(train_split_df, "New Train Split")
    check_class_distribution(val_split_df, "Validation Split")
    check_class_distribution(test_df, "Original Test")


    train_split_path = config.OUTPUT_DIR / "train_split.csv"
    val_split_path = config.OUTPUT_DIR / "val_split.csv"
    test_path_copy = config.OUTPUT_DIR / "test_split.csv"

    train_split_df.to_csv(train_split_path, index=False)
    val_split_df.to_csv(val_split_path, index=False)
    test_df.to_csv(test_path_copy, index=False)

    print(f"\n✓ Saved train split: {train_split_path}")
    print(f"✓ Saved validation split: {val_split_path}")
    print(f"✓ Saved test set copy: {test_path_copy}")


if __name__ == "__main__":
    main()
