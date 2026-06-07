import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class Config:
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
        for d in [self.EDA_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

config = Config()


def load_data():
    """Load training and testing datasets"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    print(f"\nTraining set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    return train_df, test_df


def identify_identifier_columns(df):
    identifier_candidates = []
    known_identifiers = ['srcip', 'dstip']
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in known_identifiers or 'ip' in col_lower:
            identifier_candidates.append(col)
        if col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() == len(df) and 'id' in col_lower:
                identifier_candidates.append(col)
    return list(set(identifier_candidates))

def identify_high_missing_columns(df, threshold=0.40):
    missing_pct = (df.isnull().sum() / len(df))
    high_missing = missing_pct[missing_pct > threshold].index.tolist()
    if high_missing:
        print(f"\nColumns with >{threshold*100}% missing:")
        for col in high_missing:
            print(f"  - {col}: {missing_pct[col]*100:.1f}% missing")
    else:
        print(f"\nNo columns exceed {threshold*100}% missing threshold.")
    return high_missing

def identify_constant_columns(df, numeric_var_threshold=0.001):
    constant_cols = []
    print("\nChecking for constant/near-constant columns...")


    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
            print(f"  - {col}: only 1 unique value (categorical)")


    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [config.LABEL_COL]]
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
            print(f"  - {col}: only 1 unique value (numeric)")
            continue
        col_std = df[col].std()
        normalized_var = (df[col].var() / (col_std ** 2)) if col_std > 0 else 0
        if normalized_var < numeric_var_threshold:
            constant_cols.append(col)
            print(f"  - {col}: near-constant, variance={normalized_var:.6f}")

    if not constant_cols:
        print("  None found")
    return list(set(constant_cols))

def drop_features(df, cols_to_drop, reason):
    original_shape = df.shape
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if not cols_to_drop:
        return df, {'reason': reason, 'columns': [], 'n_dropped': 0}
    df = df.drop(columns=cols_to_drop)
    log_entry = {
        'reason': reason,
        'columns': cols_to_drop,
        'n_dropped': len(cols_to_drop),
        'shape_before': original_shape,
        'shape_after': df.shape
    }
    print(f"\n✓ Dropped {len(cols_to_drop)} columns ({reason})")
    print(f"  Shape: {original_shape} -> {df.shape}")
    return df, log_entry


def main():
    print("\n" + "="*80)
    print("UNSW-NB15 PREPROCESSING - PART 2: DROP BAD FEATURES")
    print("="*80)

    train_df, test_df = load_data()
    print(f"Original shapes - Train: {train_df.shape}, Test: {test_df.shape}")

    drop_log = {'train': [], 'test': []}


    print("\n" + "="*80)
    print("STEP 1: IDENTIFYING IDENTIFIER COLUMNS")
    print("="*80)
    identifier_cols = identify_identifier_columns(train_df)
    identifiers_to_drop = [col for col in identifier_cols if 'ip' in col.lower()]
    print(f"Identified identifiers to drop: {identifiers_to_drop}")


    print("\n" + "="*80)
    print("STEP 2: HIGH MISSING VALUE COLUMNS")
    print("="*80)
    high_missing_cols = identify_high_missing_columns(train_df, threshold=config.MISSING_THRESHOLD)


    print("\n" + "="*80)
    print("STEP 3: CONSTANT/NEAR-CONSTANT FEATURES")
    print("="*80)
    constant_cols = identify_constant_columns(train_df, numeric_var_threshold=config.VARIANCE_THRESHOLD)


    all_dropped_cols = list(set(identifiers_to_drop + high_missing_cols + constant_cols))
    print("\nDropping from TRAIN dataset...")
    train_df, log1 = drop_features(train_df, all_dropped_cols, "train_drops")
    drop_log['train'].append(log1)


    print("\nApplying same drops to TEST dataset...")
    test_df, log2 = drop_features(test_df, all_dropped_cols, "same_as_train")
    drop_log['test'].append(log2)


    train_clean_path = config.OUTPUT_DIR / "train_cleaned.csv"
    test_clean_path = config.OUTPUT_DIR / "test_cleaned.csv"
    train_df.to_csv(train_clean_path, index=False)
    test_df.to_csv(test_clean_path, index=False)
    print(f"✓ Saved cleaned train -> {train_clean_path}")
    print(f"✓ Saved cleaned test  -> {test_clean_path}")

    drop_log['summary'] = {
        'total_dropped': len(all_dropped_cols),
        'columns_dropped': all_dropped_cols,
        'final_train_shape': train_df.shape,
        'final_test_shape': test_df.shape
    }
    with open(config.LOGS_DIR / "feature_drop_log.json", "w") as f:
        json.dump(drop_log, f, indent=2)
    print("✓ Saved feature_drop_log.json")

    print("\n" + "="*80)
    print("PART 2 COMPLETE ")
    print("="*80)
    print(f"Total columns dropped: {len(all_dropped_cols)}")
    print(f"Final shapes -> Train: {train_df.shape}, Test: {test_df.shape}")
    print("Next step: Run part3_train_val_split.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
