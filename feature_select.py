import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class Config:
    OUTPUT_DIR = Path("preprocessing_outputs")
    MODELS_DIR = OUTPUT_DIR / "models"
    LOGS_DIR = OUTPUT_DIR / "logs"
    EDA_DIR = OUTPUT_DIR / "eda"

    LABEL_COL = "label"
    ATTACK_CAT_COL = "attack_cat"

    RF_N_FEATURES = 40
    MI_N_FEATURES = 40

    MIN_FEATURES = 10
    TARGET_FEATURES = 30
    MAX_FEATURES = 40
    GAN_FEATURES = 4
    MAX_FEATURE_CORR = 0.6

    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10

    L1_C = 0.1

    RANDOM_STATE = 42

config = Config()


def load_imputed_data():
    """Load imputed datasets from Part 4"""
    print("Loading imputed datasets...")

    train_df = pd.read_csv(config.OUTPUT_DIR / "train_imputed.csv")
    val_df = pd.read_csv(config.OUTPUT_DIR / "val_imputed.csv")
    test_df = pd.read_csv(config.OUTPUT_DIR / "test_imputed.csv")

    print(f"Train: {train_df.shape}")
    print(f"Val: {val_df.shape}")
    print(f"Test: {test_df.shape}")

    return train_df, val_df, test_df

def prepare_features_for_selection(df):
    """Prepare features for selection algorithms"""
    if config.LABEL_COL not in df.columns:
        raise ValueError(f"Target column '{config.LABEL_COL}' not found!")

    y = df[config.LABEL_COL].values

    feature_cols = [col for col in df.columns
                   if col not in [config.LABEL_COL, config.ATTACK_CAT_COL]]

    X = df[feature_cols].copy()

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    print(f"\nFeature preparation:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Numeric: {len(feature_cols) - len(categorical_cols)}")
    print(f"  Categorical (encoded): {len(categorical_cols)}")
    print(f"  Target classes: {np.unique(y)}")

    return X, y, feature_cols, categorical_cols, label_encoders

def random_forest_selection(X, y, n_features=40):


    print(f"\nTraining Random Forest ({config.RF_N_ESTIMATORS} trees)...")

    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )

    rf.fit(X, y)

    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 features by RF importance:")
    print(importances.head(10).to_string(index=False))

    top_features = importances.head(n_features)['feature'].tolist()

    print(f"\n✓ Selected top {len(top_features)} features from Random Forest")

    plt.figure(figsize=(12, 8))
    top_20 = importances.head(20)
    plt.barh(range(len(top_20)), top_20['importance'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Features by Random Forest Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(config.EDA_DIR / "rf_feature_importance.png", dpi=150)
    plt.close()

    importances.to_csv(config.LOGS_DIR / "rf_feature_rankings.csv", index=False)

    return top_features, importances, rf

def mutual_information_selection(X, y, n_features=40):


    mi_scores = mutual_info_classif(
        X, y,
        discrete_features='auto',
        random_state=config.RANDOM_STATE,
        n_neighbors=3
    )

    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    print(f"\nTop 10 features by MI score:")
    print(mi_df.head(10).to_string(index=False))

    top_features = mi_df.head(n_features)['feature'].tolist()

    print(f"\n✓ Selected top {len(top_features)} features from Mutual Information")

    plt.figure(figsize=(12, 8))
    top_20 = mi_df.head(20)
    plt.barh(range(len(top_20)), top_20['mi_score'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Mutual Information Score')
    plt.title('Top 20 Features by Mutual Information')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(config.EDA_DIR / "mi_feature_scores.png", dpi=150)
    plt.close()

    mi_df.to_csv(config.LOGS_DIR / "mi_feature_rankings.csv", index=False)

    return top_features, mi_df

def l1_logistic_selection(X, y, C=0.1):


    lr = LogisticRegression(
        penalty='l1',
        C=C,
        solver='liblinear',
        random_state=config.RANDOM_STATE,
        class_weight='balanced',
        max_iter=1000
    )

    lr.fit(X, y)

    coefficients = pd.DataFrame({
        'feature': X.columns,
        'coefficient': np.abs(lr.coef_[0])
    }).sort_values('coefficient', ascending=False)

    non_zero = coefficients[coefficients['coefficient'] > 0]

    print(f"\nNon-zero coefficients: {len(non_zero)} / {len(X.columns)}")
    print(f"\nTop 10 features by |coefficient|:")
    print(non_zero.head(10).to_string(index=False))

    top_features = non_zero['feature'].tolist()

    print(f"\n Selected {len(top_features)} features with non-zero coefficients")

    plt.figure(figsize=(12, 8))
    top_20 = non_zero.head(20)
    plt.barh(range(len(top_20)), top_20['coefficient'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Top 20 Features by L1-Logistic Regression')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(config.EDA_DIR / "l1_feature_coefficients.png", dpi=150)
    plt.close()

    non_zero.to_csv(config.LOGS_DIR / "l1_feature_rankings.csv", index=False)

    return top_features, non_zero, lr

def combine_selections(rf_features, mi_features, l1_features, all_feature_cols):


    rf_set = set(rf_features)
    mi_set = set(mi_features)
    l1_set = set(l1_features)

    intersection = rf_set & mi_set & l1_set
    union = rf_set | mi_set | l1_set

    majority = set()
    for feature in union:
        count = sum([feature in rf_set, feature in mi_set, feature in l1_set])
        if count >= 2:
            majority.add(feature)

    print(f"\nMethod overlap:")
    print(f"  RF selected: {len(rf_set)}")
    print(f"  MI selected: {len(mi_set)}")
    print(f"  L1 selected: {len(l1_set)}")
    print(f"  Intersection (all 3): {len(intersection)}")
    print(f"  Majority (≥2): {len(majority)}")
    print(f"  Union (any): {len(union)}")

    print(f"\nBuilding final selection (target: {config.TARGET_FEATURES})...")

    selected_features = list(intersection)
    print(f"  Start with intersection: {len(selected_features)}")

    if len(selected_features) < config.TARGET_FEATURES:
        additional = majority - intersection
        selected_features.extend(list(additional))
        print(f"  Added majority vote: {len(selected_features)}")

    if len(selected_features) < config.TARGET_FEATURES:
        remaining = union - set(selected_features)
        scored = [(f, sum([f in rf_set, f in mi_set, f in l1_set]))
                  for f in remaining]
        scored.sort(key=lambda x: x[1], reverse=True)

        needed = min(config.TARGET_FEATURES - len(selected_features), len(scored))
        selected_features.extend([f for f, _ in scored[:needed]])
        print(f"  Added top union features: {len(selected_features)}")

    if len(selected_features) > config.MAX_FEATURES:
        scored = [(f, sum([f in rf_set, f in mi_set, f in l1_set]))
                  for f in selected_features]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in scored[:config.MAX_FEATURES]]
        print(f"  Capped at max: {len(selected_features)}")

    print(f"\n Final selection: {len(selected_features)} features")

    selection_details = []
    for feature in selected_features:
        details = {
            'feature': feature,
            'in_rf': feature in rf_set,
            'in_mi': feature in mi_set,
            'in_l1': feature in l1_set,
            'selection_count': sum([feature in rf_set, feature in mi_set, feature in l1_set])
        }
        selection_details.append(details)

    details_df = pd.DataFrame(selection_details).sort_values('selection_count', ascending=False)


    return selected_features, details_df

def save_selection_results(selected_features, details_df, rf_model, lr_model):
    selection_dir = config.MODELS_DIR / "feature_selection"
    selection_dir.mkdir(exist_ok=True)

    features_path = selection_dir / "selected_features.json"
    with open(features_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f"\n Saved selected features: {features_path}")

    top_4_path = selection_dir / "top_4_features.json"
    with open(top_4_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f" Saved top 4 GAN features: {top_4_path}")

    details_path = selection_dir / "selection_details.csv"
    details_df.to_csv(details_path, index=False)
    print(f" Saved selection details: {details_path}")

    with open(selection_dir / "rf_model.pkl", 'wb') as f:
        pickle.dump(rf_model, f)

    with open(selection_dir / "lr_model.pkl", 'wb') as f:
        pickle.dump(lr_model, f)


    return selection_dir

def rank_features_for_gan(selected_features, rf_importances, mi_scores, l1_coeffs):
    """Create a stable ensemble ranking before reducing to four GAN features."""
    rf_df = rf_importances[["feature", "importance"]].copy()
    mi_df = mi_scores[["feature", "mi_score"]].copy()
    l1_df = l1_coeffs[["feature", "coefficient"]].copy()

    for df, score_col, norm_col in [
        (rf_df, "importance", "rf_norm"),
        (mi_df, "mi_score", "mi_norm"),
        (l1_df, "coefficient", "l1_norm"),
    ]:
        score_min = df[score_col].min()
        score_max = df[score_col].max()
        if score_max > score_min:
            df[norm_col] = (df[score_col] - score_min) / (score_max - score_min)
        else:
            df[norm_col] = 1.0

    ranked = pd.DataFrame({"feature": selected_features})
    ranked = ranked.merge(rf_df[["feature", "rf_norm"]], on="feature", how="left")
    ranked = ranked.merge(mi_df[["feature", "mi_norm"]], on="feature", how="left")
    ranked = ranked.merge(l1_df[["feature", "l1_norm"]], on="feature", how="left")
    ranked[["rf_norm", "mi_norm", "l1_norm"]] = ranked[["rf_norm", "mi_norm", "l1_norm"]].fillna(0.0)
    ranked["ensemble_score"] = (
        0.35 * ranked["rf_norm"]
        + 0.35 * ranked["mi_norm"]
        + 0.30 * ranked["l1_norm"]
    )

    return ranked.sort_values("ensemble_score", ascending=False)["feature"].tolist()

def main():


    train_df, val_df, test_df = load_imputed_data()


    X_train, y_train, feature_cols, categorical_cols, label_encoders = prepare_features_for_selection(train_df)


    rf_features, rf_importances, rf_model = random_forest_selection(
        X_train, y_train,
        n_features=config.RF_N_FEATURES
    )


    mi_features, mi_scores = mutual_information_selection(
        X_train, y_train,
        n_features=config.MI_N_FEATURES
    )


    l1_features, l1_coeffs, lr_model = l1_logistic_selection(
        X_train, y_train,
        C=config.L1_C
    )


    candidate_features, candidate_details_df = combine_selections(
        rf_features, mi_features, l1_features, feature_cols
    )

    ranked_candidates = rank_features_for_gan(
        candidate_features, rf_importances, mi_scores, l1_coeffs
    )
    selected_features = pick_top_k_diverse(
        ranked_candidates,
        X_train,
        k=config.GAN_FEATURES,
        max_corr=config.MAX_FEATURE_CORR,
    )

    if len(selected_features) < config.GAN_FEATURES:
        selected_set = set(selected_features)
        for feature in ranked_candidates:
            if feature not in selected_set:
                selected_features.append(feature)
                selected_set.add(feature)
            if len(selected_features) == config.GAN_FEATURES:
                break

    details_df = candidate_details_df[
        candidate_details_df["feature"].isin(selected_features)
    ].copy()
    details_df["gan_rank"] = details_df["feature"].map(
        {feature: i + 1 for i, feature in enumerate(selected_features)}
    )
    details_df = details_df.sort_values("gan_rank")

    selection_dir = save_selection_results(selected_features, details_df, rf_model, lr_model)


    print(f"\nOriginal features: {len(feature_cols)}")
    print(f"Candidate features: {len(candidate_features)}")
    print(f"Selected GAN features: {len(selected_features)}")
    print(f"Reduction: {(1 - len(selected_features)/len(feature_cols))*100:.1f}%")

    print(f"\nSelected features:")
    for i, feature in enumerate(sorted(selected_features), 1):
        methods = []
        if feature in rf_features:
            methods.append("RF")
        if feature in mi_features:
            methods.append("MI")
        if feature in l1_features:
            methods.append("L1")
        print(f"  {i:2d}. {feature:30s} [{', '.join(methods)}]")

def pick_top_k_diverse(top_list, X, k=4, max_corr=0.6, banned=("id",)):
    corr = X.corr(numeric_only=True).abs().fillna(0.0)
    picked = []
    for f in top_list:
        if f in banned:
            continue
        if f not in corr.columns:
            continue
        if not picked:
            picked.append(f);
            continue
        if all(corr.loc[f, p] < max_corr for p in picked):
            picked.append(f)
        if len(picked) == k:
            break
    return picked

if __name__ == "__main__":
    main()
