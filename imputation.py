import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import warnings
warnings.filterwarnings('ignore')

class Config:
    OUTPUT_DIR = Path("preprocessing_outputs")
    MODELS_DIR = OUTPUT_DIR / "models"
    LOGS_DIR = OUTPUT_DIR / "logs"

    LABEL_COL = "label"
    ATTACK_CAT_COL = "attack_cat"

    NUMERIC_STRATEGY = 'median'
    CATEGORICAL_MISSING_TOKEN = '__MISSING__'
    USE_ITERATIVE_IMPUTER = False

    RANDOM_STATE = 42

config = Config()


def load_split_data():
    """Load train/val/test splits from Part 3"""
    print("Loading split datasets...")

    train_df = pd.read_csv(config.OUTPUT_DIR / "train_split.csv")
    val_df = pd.read_csv(config.OUTPUT_DIR / "val_split.csv")
    test_df = pd.read_csv(config.OUTPUT_DIR / "test_split.csv")

    print(f"Train: {train_df.shape}")
    print(f"Val: {val_df.shape}")
    print(f"Test: {test_df.shape}")

    return train_df, val_df, test_df

def analyze_missing_values(train_df, val_df, test_df):
    """Analyze missing value patterns across all sets"""
    print("\n" + "="*80)
    print("MISSING VALUE ANALYSIS")
    print("="*80)

    def get_missing_info(df, name):
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'column': missing.index,
            f'{name}_missing': missing.values,
            f'{name}_pct': missing_pct.values
        })
        return missing_df[missing_df[f'{name}_missing'] > 0]

    train_missing = get_missing_info(train_df, 'train')
    val_missing = get_missing_info(val_df, 'val')
    test_missing = get_missing_info(test_df, 'test')

    all_missing_cols = set(train_missing['column'].tolist() +
                          val_missing['column'].tolist() +
                          test_missing['column'].tolist())

    if len(all_missing_cols) == 0:
        print("\n✓ No missing values found in any dataset!")
        print("  Skipping imputation step.")
        return None, []

    print(f"\nColumns with missing values: {len(all_missing_cols)}")
    print(f"  {sorted(all_missing_cols)}")

    for col in sorted(all_missing_cols):
        train_miss = train_df[col].isnull().sum()
        val_miss = val_df[col].isnull().sum()
        test_miss = test_df[col].isnull().sum()

        print(f"\n{col}:")
        print(f"  Train: {train_miss:,} ({train_miss/len(train_df)*100:.2f}%)")
        print(f"  Val: {val_miss:,} ({val_miss/len(val_df)*100:.2f}%)")
        print(f"  Test: {test_miss:,} ({test_miss/len(test_df)*100:.2f}%)")

    return all_missing_cols, list(all_missing_cols)

def separate_column_types(df, missing_cols):
    """Separate features into numeric and categorical for different imputation"""
    numeric_missing = []
    categorical_missing = []

    for col in missing_cols:
        if col in [config.LABEL_COL, config.ATTACK_CAT_COL]:
            continue

        if df[col].dtype in ['int64', 'float64']:
            numeric_missing.append(col)
        else:
            categorical_missing.append(col)

    print(f"\nFeature types with missing values:")
    print(f"  Numeric: {len(numeric_missing)} features")
    if numeric_missing:
        print(f"    {numeric_missing}")
    print(f"  Categorical: {len(categorical_missing)} features")
    if categorical_missing:
        print(f"    {categorical_missing}")

    return numeric_missing, categorical_missing

def create_numeric_imputer(strategy='median', use_iterative=False):
    """Create imputer for numeric features"""
    if use_iterative:
        print(f"\nCreating IterativeImputer(random_state=config.RANDOM_STATE)...")
        print("  This models feature relationships for better imputation")
        print("  (slower but more accurate)")
        imputer = IterativeImputer(
            random_state=config.RANDOM_STATE,
            max_iter=10,
            verbose=0
        )
    else:
        print(f"\nCreating SimpleImputer(strategy='{strategy}')...")
        imputer = SimpleImputer(strategy=strategy)

    return imputer

def impute_numeric_features(train_df, val_df, test_df, numeric_cols, imputer):
    """Impute numeric features"""
    if len(numeric_cols) == 0:
        print("\n✓ No numeric features to impute")
        return train_df, val_df, test_df, None

    print(f"\nImputing {len(numeric_cols)} numeric features...")

    print("  [1/4] Fitting imputer on TRAIN set...")
    imputer.fit(train_df[numeric_cols])

    print("  [2/4] Transforming TRAIN set...")
    train_df[numeric_cols] = imputer.transform(train_df[numeric_cols])

    print("  [3/4] Transforming VAL set...")
    val_df[numeric_cols] = imputer.transform(val_df[numeric_cols])

    print("  [4/4] Transforming TEST set...")
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])

    train_missing = train_df[numeric_cols].isnull().sum().sum()
    val_missing = val_df[numeric_cols].isnull().sum().sum()
    test_missing = test_df[numeric_cols].isnull().sum().sum()

    print(f"\n Numeric imputation complete")
    print(f"  Remaining missing - Train: {train_missing}, Val: {val_missing}, Test: {test_missing}")

    if train_missing + val_missing + test_missing > 0:
        print("   WARNING: Some missing values remain!")

    return train_df, val_df, test_df, imputer

def impute_categorical_features(train_df, val_df, test_df, categorical_cols, missing_token='__MISSING__'):
    """Impute categorical features using a special missing token"""
    if len(categorical_cols) == 0:
        print("\n No categorical features to impute")
        return train_df, val_df, test_df, None

    print(f"\nImputing {len(categorical_cols)} categorical features...")
    print(f"  Strategy: Fill with '{missing_token}'")

    print("\n  Original unique values:")
    for col in categorical_cols[:3]:
        print(f"    {col}: {train_df[col].nunique()} unique")

    for col in categorical_cols:
        train_df[col] = train_df[col].fillna(missing_token)
        val_df[col] = val_df[col].fillna(missing_token)
        test_df[col] = test_df[col].fillna(missing_token)

    train_missing = train_df[categorical_cols].isnull().sum().sum()
    val_missing = val_df[categorical_cols].isnull().sum().sum()
    test_missing = test_df[categorical_cols].isnull().sum().sum()

    print(f"\n Categorical imputation complete")
    print(f"  Remaining missing - Train: {train_missing}, Val: {val_missing}, Test: {test_missing}")

    imputer_info = {
        'strategy': 'constant',
        'fill_value': missing_token,
        'columns': categorical_cols
    }

    return train_df, val_df, test_df, imputer_info

def save_imputers(numeric_imputer, categorical_imputer_info):
    imputers_dir = config.MODELS_DIR / "imputers"
    imputers_dir.mkdir(exist_ok=True)

    saved_files = []

    if numeric_imputer is not None:
        numeric_path = imputers_dir / "numeric_imputer.pkl"
        with open(numeric_path, 'wb') as f:
            pickle.dump(numeric_imputer, f)
        print(f"\n Saved numeric imputer: {numeric_path}")
        saved_files.append(str(numeric_path))

    if categorical_imputer_info is not None:
        categorical_path = imputers_dir / "categorical_imputer.json"
        with open(categorical_path, 'w') as f:
            json.dump(categorical_imputer_info, f, indent=2)
        print(f"✓ Saved categorical imputer info: {categorical_path}")
        saved_files.append(str(categorical_path))

    metadata = {
        'numeric_strategy': config.NUMERIC_STRATEGY,
        'use_iterative': config.USE_ITERATIVE_IMPUTER,
        'categorical_token': config.CATEGORICAL_MISSING_TOKEN,
        'saved_files': saved_files
    }

    metadata_path = imputers_dir / "imputation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved imputation metadata: {metadata_path}")

    return imputers_dir

def verify_imputation(train_df, val_df, test_df):

    train_missing = train_df.isnull().sum().sum()
    val_missing = val_df.isnull().sum().sum()
    test_missing = test_df.isnull().sum().sum()

    print(f"\nTotal missing values:")
    print(f"  Train: {train_missing}")
    print(f"  Val: {val_missing}")
    print(f"  Test: {test_missing}")

    if train_missing + val_missing + test_missing == 0:
        print("\n SUCCESS: All missing values imputed!")
        return True
    else:
        print("\n WARNING: Missing values still present!")

        for df, name in [(train_df, 'Train'), (val_df, 'Val'), (test_df, 'Test')]:
            still_missing = df.isnull().sum()
            still_missing = still_missing[still_missing > 0]
            if len(still_missing) > 0:
                print(f"\n{name} - columns with missing:")
                print(still_missing)

        return False

def main():

    print(f"  Numeric: {config.NUMERIC_STRATEGY}")
    print(f"  Iterative: {config.USE_ITERATIVE_IMPUTER}")
    print(f"  Categorical: Fill with '{config.CATEGORICAL_MISSING_TOKEN}'")
    print("="*80)


    train_df, val_df, test_df = load_split_data()


    all_missing_cols, missing_cols_list = analyze_missing_values(train_df, val_df, test_df)

    if all_missing_cols is None or len(missing_cols_list) == 0:
        print("\n" + "="*80)
        print("NO IMPUTATION NEEDED - Saving datasets as-is")
        print("="*80)

        train_df.to_csv(config.OUTPUT_DIR / "train_imputed.csv", index=False)
        val_df.to_csv(config.OUTPUT_DIR / "val_imputed.csv", index=False)
        test_df.to_csv(config.OUTPUT_DIR / "test_imputed.csv", index=False)

        print("\n✓ Datasets saved (no changes made)")
        print("\nNext step: Run part5_feature_selection.py")
        print("="*80 + "\n")
        return


    numeric_cols, categorical_cols = separate_column_types(train_df, missing_cols_list)


    numeric_imputer = create_numeric_imputer(
        strategy=config.NUMERIC_STRATEGY,
        use_iterative=config.USE_ITERATIVE_IMPUTER
    )

    train_df, val_df, test_df, numeric_imputer = impute_numeric_features(
        train_df, val_df, test_df,
        numeric_cols,
        numeric_imputer
    )


    train_df, val_df, test_df, categorical_imputer_info = impute_categorical_features(
        train_df, val_df, test_df,
        categorical_cols,
        missing_token=config.CATEGORICAL_MISSING_TOKEN
    )


    train_path = config.OUTPUT_DIR / "train_imputed.csv"
    val_path = config.OUTPUT_DIR / "val_imputed.csv"
    test_path = config.OUTPUT_DIR / "test_imputed.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n✓ Saved imputed datasets:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

    imputers_dir = save_imputers(numeric_imputer, categorical_imputer_info)

    imputation_log = {
        'numeric_features_imputed': numeric_cols,
        'categorical_features_imputed': categorical_cols,
        'numeric_strategy': config.NUMERIC_STRATEGY,
        'categorical_strategy': f"constant ('{config.CATEGORICAL_MISSING_TOKEN}')",
        'use_iterative': config.USE_ITERATIVE_IMPUTER,
        'train_shape': train_df.shape,
        'val_shape': val_df.shape,
        'test_shape': test_df.shape
    }

    log_path = config.LOGS_DIR / "imputation_log.json"
    with open(log_path, 'w') as f:
        json.dump(imputation_log, f, indent=2)

    print(f"✓ Saved imputation log: {log_path}")

    print(f"  Val: {val_df.shape}")
    print(f"  Test: {test_df.shape}")


if __name__ == "__main__":
    main()
