"""
Meta-Learner training with Gene-Disjoint Split
Loads pre-split train/test from splits/gene_disjoint_{train,test}.tsv
Also evaluates on D2 as external temporal test.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, accuracy_score, roc_curve
from scipy.stats import uniform, randint


EXCLUDE_COLS = {'rcv_accession', 'gene_symbol', 'label', 'label_binary'}


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def evaluate(y_true, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, (y_prob >= 0.5).astype(int))
    f1 = f1_score(y_true, (y_prob >= 0.5).astype(int))
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    threshold = thresholds[opt_idx]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        'AUROC': auroc, 'AUPRC': auprc, 'MCC': mcc, 'F1': f1, 'Accuracy': acc,
        'TP': int(np.sum((y_pred==1) & (y_true==1))),
        'FP': int(np.sum((y_pred==1) & (y_true==0))),
        'FN': int(np.sum((y_pred==0) & (y_true==1))),
        'TN': int(np.sum((y_pred==0) & (y_true==0))),
        'Optimal_Threshold': float(threshold),
    }


def train_meta_gene_split(dataset_dir, d2_dir=None, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, 'meta_model_gene_split')
    os.makedirs(output_dir, exist_ok=True)

    split_dir = os.path.join(dataset_dir, 'splits')
    train_path = os.path.join(split_dir, 'gene_disjoint_train.tsv')
    test_path = os.path.join(split_dir, 'gene_disjoint_test.tsv')

    print(f"Loading gene-disjoint splits...")
    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')

    feature_cols = get_feature_cols(train_df)
    print(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")

    X_train = train_df[feature_cols].copy()
    y_train = train_df['label_binary'].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df['label_binary'].copy()

    print(f"\nTrain: {len(X_train)} variants ({int(y_train.sum())} path, {int((y_train==0).sum())} ben)")
    print(f"Test:  {len(X_test)} variants ({int(y_test.sum())} path, {int((y_test==0).sum())} ben)")

    print("\n=== Hyperparameter Tuning (RandomizedSearchCV, 80 iterations, 5-fold CV) ===")
    base_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        tree_method='hist', device='cuda', verbosity=0, random_state=42,
    )

    param_distributions = {
        'max_depth': randint(3, 11),
        'learning_rate': uniform(0.01, 0.29),
        'n_estimators': randint(200, 1600),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.5, 0.5),
        'min_child_weight': randint(1, 21),
        'reg_alpha': [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1, 5, 10],
        'reg_lambda': [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1, 5, 10],
        'gamma': uniform(0, 5),
    }

    search = RandomizedSearchCV(
        base_model, param_distributions, n_iter=80,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc', random_state=42, n_jobs=1, verbose=2,
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print(f"\nBest CV AUROC: {search.best_score_:.6f}")
    print(f"Best params: {best_params}")

    final_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'tree_method': 'hist', 'device': 'cuda', 'verbosity': 0, 'random_state': 42,
        **best_params,
    }

    print("\n=== 5-Fold Cross-Validation (Train Set) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train)):
        Xtr, Xvl = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
        ytr, yvl = y_train.iloc[tr_idx], y_train.iloc[vl_idx]
        model = xgb.XGBClassifier(**final_params)
        model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
        y_prob = model.predict_proba(Xvl)[:, 1]
        metrics = evaluate(yvl.values, y_prob)
        cv_results.append(metrics)
        print(f"  Fold {fold+1}: AUROC={metrics['AUROC']:.4f}  MCC={metrics['MCC']:.4f}")

    cv_summary = {
        'AUROC_mean': float(np.mean([r['AUROC'] for r in cv_results])),
        'AUROC_std': float(np.std([r['AUROC'] for r in cv_results])),
        'MCC_mean': float(np.mean([r['MCC'] for r in cv_results])),
        'MCC_std': float(np.std([r['MCC'] for r in cv_results])),
    }
    print(f"\nCV: AUROC={cv_summary['AUROC_mean']:.4f} +/- {cv_summary['AUROC_std']:.4f}")

    print("\n=== Final Model on Full Train Set ===")
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test.values, y_test_prob)
    print(f"\n=== Gene-Disjoint Test Set Results ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    individual_models = {
        'ESM1b-MLM': 'esm1b_mlm', 'ESM-1v-MLM': 'esm1v_mlm',
        'ESM2-MLM': 'esm2_mlm', 'ESM3-MLM': 'esm3_mlm',
        'ProtBERT-MLM': 'protbert_mlm', 'NT-v2-MLM': 'ntv2_mlm',
        'ProtBERT-Cos': 'protbert_cos', 'NT-v2-Cos': 'ntv2_cos',
    }
    individual_results = {}
    print("\n=== Individual Models (Gene-Disjoint Test) ===")
    for name, col in individual_models.items():
        if col in test_df.columns:
            try:
                auroc = roc_auc_score(y_test, test_df[col].values)
                individual_results[name] = auroc
                print(f"  {name}: AUROC={auroc:.4f}")
            except:
                pass
    individual_results['Meta-Learner (Ours)'] = test_metrics['AUROC']
    print(f"  Meta-Learner (Ours): AUROC={test_metrics['AUROC']:.4f}")

    d2_results = {}
    if d2_dir and os.path.exists(os.path.join(d2_dir, 'feature_matrix.tsv')):
        print(f"\n=== D2 Temporal External Test ===")
        d2_df = pd.read_csv(os.path.join(d2_dir, 'feature_matrix.tsv'), sep='\t')
        X_d2 = d2_df[feature_cols].copy()
        y_d2 = d2_df['label_binary'].copy()

        y_d2_prob = final_model.predict_proba(X_d2)[:, 1]
        d2_metrics = evaluate(y_d2.values, y_d2_prob)
        print(f"  Meta-Learner on D2: AUROC={d2_metrics['AUROC']:.4f}")

        for name, col in individual_models.items():
            if col in d2_df.columns:
                try:
                    auroc = roc_auc_score(y_d2, d2_df[col].values)
                    d2_results[name] = auroc
                    print(f"  {name}: AUROC={auroc:.4f}")
                except:
                    pass
        d2_results['Meta-Learner (Ours)'] = d2_metrics['AUROC']
        d2_results_json = {k: float(v) for k, v in d2_results.items()}
    else:
        d2_metrics = None
        d2_results_json = {}

    importance = final_model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print(f"\nTop 15 Features:")
    for _, row in feat_imp.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    test_info = test_df[['rcv_accession', 'gene_symbol', 'label']].copy()
    test_info['meta_score'] = y_test_prob
    test_info['label_binary'] = y_test.values
    test_info.to_csv(os.path.join(output_dir, 'test_predictions.tsv'), sep='\t', index=False)
    feat_imp.to_csv(os.path.join(output_dir, 'feature_importance.tsv'), sep='\t', index=False)

    results = {
        'split': 'gene_disjoint',
        'train_genes': int(train_df['gene_symbol'].nunique()),
        'test_genes': int(test_df['gene_symbol'].nunique()),
        'gene_overlap': 0,
        'best_params': {k: float(v) if isinstance(v, (float, np.floating, np.integer)) else v for k, v in best_params.items()},
        'cv_summary': cv_summary,
        'test_metrics': {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in test_metrics.items()},
        'individual_aurocs': {k: float(v) for k, v in individual_results.items()},
        'd2_individual_aurocs': d2_results_json,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(final_model, f)

    print(f"\nSaved to {output_dir}/")
    return results


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else 'dataset1_clinvar_only'
    d2 = sys.argv[2] if len(sys.argv) > 2 else 'dataset2_jan2025'
    train_meta_gene_split(ds, d2)
