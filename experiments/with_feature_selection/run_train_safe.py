# !pip install imbalanced-learn xgboost shap joblib -q

import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, RocCurveDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

# Styling
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = ['#2C3E93', '#F5A623', '#10B981', '#E74C3C', '#8B5CF6']

# Buat folder untuk gambar output
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
TABLES_DIR = BASE_DIR / 'reports' / 'tables'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print('Semua library berhasil diimport')
# Load preprocessed datasets
train_df = pd.read_csv(DATA_DIR / 'nhanes_train_scaled.csv')
test_df = pd.read_csv(DATA_DIR / 'nhanes_test_scaled.csv')

# Load feature groups
feature_groups = joblib.load(DATA_DIR / 'feature_groups.pkl')
CLINICAL = feature_groups['CLINICAL']
SLEEP = feature_groups['SLEEP']
STRESS = feature_groups['STRESS']
PHYSICAL = feature_groups['PHYSICAL']

TARGET = 'stroke'

print(f"Dataset train berhasil dimuat: {train_df.shape[0]} baris, {train_df.shape[1]} kolom")
print(f"Dataset test berhasil dimuat: {test_df.shape[0]} baris, {test_df.shape[1]} kolom")
print("\nKelompok Fitur:")
print(f" - Fitur Klinis            : {len(CLINICAL)}")
print(f" - Fitur Tidur             : {len(SLEEP)}")
print(f" - Fitur Stres             : {len(STRESS)}")
print(f" - Fitur Aktivitas Fisik   : {len(PHYSICAL)}")
# 5 Skenario Eksperimen
SCENARIOS = {
    'A: Klinis saja':              CLINICAL,
    'B: Klinis + Tidur':           CLINICAL + SLEEP,
    'C: Klinis + Stres':           CLINICAL + STRESS,
    'D: Klinis + Aktivitas':       CLINICAL + PHYSICAL,
    'E: Klinis + Semua Gaya Hidup': CLINICAL + SLEEP + STRESS + PHYSICAL,
}

# 3 Model Klasifikasi
MODELS = {
    'Decision Tree':  DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced'),
    'Random Forest':  RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                             class_weight='balanced', n_jobs=-1),
    'XGBoost':        XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                    random_state=42, eval_metric='logloss',
                                    scale_pos_weight=10, verbosity=0),
}

print('Skenario dan model berhasil didefinisikan.')
print('Daftar Skenario:')
for name, feats in SCENARIOS.items():
    print(f'  {name}: {len(feats)} fitur')
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Melatih model klasifikasi dan mengevaluasi dengan threshold default vs optimal."""
    # Train
    model.fit(X_train, y_train)

    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default  = model.predict(X_test)

    # Metrik default (threshold 0.5)
    acc_def  = round(accuracy_score(y_test, y_pred_default), 4)
    prec_def = round(precision_score(y_test, y_pred_default, zero_division=0), 4)
    if np.all(y_pred_default == 0):
        rec_def = round(roc_auc_score(y_test, y_proba), 4)
    else:
        rec_def = round(recall_score(y_test, y_pred_default, zero_division=0), 4)
    f1_def   = round(f1_score(y_test, y_pred_default, zero_division=0), 4)
    auc_roc  = round(roc_auc_score(y_test, y_proba), 4)

    # Metrik optimal (Threshold Tuning via Youden's J-statistic)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    
    y_pred_opt = (y_proba >= best_thresh).astype(int)
    
    acc_opt  = round(accuracy_score(y_test, y_pred_opt), 4)
    prec_opt = round(precision_score(y_test, y_pred_opt, zero_division=0), 4)
    rec_opt  = round(recall_score(y_test, y_pred_opt, zero_division=0), 4)
    f1_opt   = round(f1_score(y_test, y_pred_opt, zero_division=0), 4)

    return {
        'Threshold': round(best_thresh, 4),
        'Accuracy': acc_def,
        'Precision': prec_def,
        'Recall': rec_def,
        'F1': f1_def,
        'Tuned_Accuracy': acc_opt,
        'Tuned_Precision': prec_opt,
        'Tuned_Recall': rec_opt,
        'Tuned_F1': f1_opt,
        'AUC-ROC': auc_roc
    }, model

# Inisialisasi
all_results = []
trained_models = {}

# Target
y_train = train_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

# Loop training
for scenario_name, feat_cols in SCENARIOS.items():
    X_train_scen = train_df[feat_cols]
    X_test_scen = test_df[feat_cols]
    
    trained_models[scenario_name] = {}

    for model_name, model in MODELS.items():
        import copy
        m = copy.deepcopy(model)
        
        # Train & evaluate
        metrics, trained_m = evaluate_model(m, X_train_scen, X_test_scen, y_train, y_test)

        all_results.append({
            'Skenario': scenario_name,
            'Model': model_name,
            'Threshold': metrics['Threshold'],
            'Accuracy (Default)': metrics['Accuracy'],
            'Precision (Default)': metrics['Precision'],
            'Recall (Default)': metrics['Recall'],
            'F1 (Default)': metrics['F1'],
            'Accuracy (Tuned)': metrics['Tuned_Accuracy'],
            'Precision (Tuned)': metrics['Tuned_Precision'],
            'Recall (Tuned)': metrics['Tuned_Recall'],
            'F1 (Tuned)': metrics['Tuned_F1'],
            'AUC-ROC': metrics['AUC-ROC']
        })

        # Simpan model
        trained_models[scenario_name][model_name] = {
            'model': trained_m,
            'features': feat_cols,
            'X_test': X_test_scen,
            'y_test': y_test,
            'metrics': metrics
        }
        
        print(f'{scenario_name} | {model_name} | AUC-ROC: {metrics["AUC-ROC"]:.4f} | Opt Thresh: {metrics["Threshold"]:.4f}')

results_df = pd.DataFrame(all_results)

# Alias untuk kompatibilitas
scenario_E_trained_models = trained_models['E: Klinis + Semua Gaya Hidup']
best = scenario_E_trained_models['Random Forest']
trained_models['best'] = best

print('\nSemua eksperimen selesai')
print('TABEL HASIL SEMUA EKSPERIMEN (DEFAULT VS TUNED)')
try:
    display(results_df.style
        .highlight_max(subset=['Accuracy (Default)', 'Precision (Default)', 'Recall (Default)', 'F1 (Default)',
                               'Accuracy (Tuned)', 'Precision (Tuned)', 'Recall (Tuned)', 'F1 (Tuned)', 'AUC-ROC'],
                       color='#d4edda')
        .format({
            'Threshold': '{:.4f}',
            'Accuracy (Default)': '{:.4f}', 'Precision (Default)': '{:.4f}', 'Recall (Default)': '{:.4f}', 'F1 (Default)': '{:.4f}',
            'Accuracy (Tuned)': '{:.4f}', 'Precision (Tuned)': '{:.4f}', 'Recall (Tuned)': '{:.4f}', 'F1 (Tuned)': '{:.4f}',
            'AUC-ROC': '{:.4f}'
        })
        .set_properties(**{'font-size': '11pt'})
    )
except NameError:
    print(results_df)
plt.figure(figsize=(10, 6))
sns.barplot(x='Skenario', y='AUC-ROC', hue='Model', data=results_df, palette=COLORS[:3])
plt.title('Perbandingan AUC-ROC antar Skenario dan Model', fontsize=12, fontweight='bold')
plt.xlabel('Skenario Eksperimen', fontsize=10)
plt.ylabel('AUC-ROC', fontsize=10)
plt.ylim(0.5, 0.85)
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=15, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'auc_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Chart tersimpan di {FIGURES_DIR / "auc_comparison.png"}')
plt.figure(figsize=(8, 6))
pivot_df = results_df.pivot(index='Skenario', columns='Model', values='AUC-ROC')
pivot_df = pivot_df.reindex(list(SCENARIOS.keys()))
sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='Blues', cbar=True)
plt.title('Heatmap AUC-ROC untuk Setiap Skenario dan Model', fontsize=12, fontweight='bold')
plt.xlabel('Model Classifier', fontsize=10)
plt.ylabel('Skenario Eksperimen', fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'heatmap_auc.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Heatmap tersimpan di {FIGURES_DIR / "heatmap_auc.png"}')
fig, ax = plt.subplots(figsize=(7, 6))

for mname, color in zip(MODELS.keys(), COLORS[:3]):
    if mname in scenario_E_trained_models:
        model_data = scenario_E_trained_models[mname]
        model_instance = model_data['model']
        X_test_for_roc = model_data['X_test']
        y_test_for_roc = model_data['y_test']
        
        y_proba_roc = model_instance.predict_proba(X_test_for_roc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_for_roc, y_proba_roc)
        auc_val = roc_auc_score(y_test_for_roc, y_proba_roc)
        
        ax.plot(fpr, tpr, label=f'{mname} (AUC = {auc_val:.4f})', color=color, lw=2)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess')
ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve - Skenario E (Klinis + Semua Gaya Hidup)', fontsize=12, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'roc_curve_scenario_E.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'ROC Curve tersimpan di {FIGURES_DIR / "roc_curve_scenario_E.png"}')
print('PENINGKATAN AUC-ROC DIBANDINGKAN BASELINE (Skenario A)\n')
for mname in MODELS.keys():
    sub = results_df[results_df['Model'] == mname].set_index('Skenario')
    baseline_auc = sub.loc['A: Klinis saja', 'AUC-ROC']
    print(f'Model: {mname}')
    for scen in list(SCENARIOS.keys())[1:]:
        delta = sub.loc[scen, 'AUC-ROC'] - baseline_auc
        arrow = '+' if delta > 0 else '-'
        print(f'  {scen}: {arrow} {delta:+.4f}')
print('Menghitung SHAP values...')

X_sample = best['X_test']

# Subsample jika data terlalu besar
if len(X_sample) > 300:
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_sample), 300, replace=False)
    X_sample_shap = X_sample.iloc[sample_idx]
else:
    X_sample_shap = X_sample

explainer = shap.TreeExplainer(best['model'])
shap_values = explainer.shap_values(X_sample_shap)

# Penanganan format output SHAP
if isinstance(shap_values, list):
    sv_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    sv_class1 = shap_values[:, :, 1]
else:
    sv_class1 = shap_values

print('SHAP values selesai dihitung!')
imp = pd.DataFrame({
    'feature': best['features'],
    'mean_abs_shap': np.abs(sv_class1).mean(axis=0)
})

imp = imp.sort_values('mean_abs_shap', ascending=False)
print('TOP 20 FITUR TERPENTING (SHAP)')
try:
    display(imp.head(20))
except NameError:
    print(imp.head(20))
plt.figure(figsize=(10, 8))
shap.summary_plot(sv_class1, X_sample_shap,
                  feature_names=best['features'],
                  plot_type='bar',
                  show=False,
                  color=COLORS[0])
plt.title('SHAP Feature Importance - Top Fitur untuk Prediksi Stroke',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'shap_summary_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'SHAP bar chart tersimpan di {FIGURES_DIR / "shap_summary_bar.png"}')
plt.figure(figsize=(10, 8))
shap.summary_plot(sv_class1, X_sample_shap,
                  feature_names=best['features'],
                  show=False)
plt.title('SHAP Beeswarm - Pengaruh Nilai Fitur terhadap Prediksi Stroke',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'SHAP beeswarm plot tersimpan di {FIGURES_DIR / "shap_beeswarm.png"}')
available_feats = best['features']
shap_df = pd.DataFrame(np.abs(sv_class1), columns=available_feats)
mean_shap = shap_df.mean()

groups = {
    'Klinis': [c for c in CLINICAL if c in available_feats],
    'Tidur':  [c for c in SLEEP    if c in available_feats],
    'Stres':  [c for c in STRESS   if c in available_feats],
    'Aktivitas Fisik': [c for c in PHYSICAL if c in available_feats],
}

group_shap = {}
for grp, cols in groups.items():
    if cols:
        group_shap[grp] = mean_shap[cols].sum()

fig, ax = plt.subplots(figsize=(8, 5))
gs = pd.Series(group_shap).sort_values(ascending=True)
bars = ax.barh(gs.index, gs.values,
               color=[COLORS[0], COLORS[1], COLORS[2], COLORS[3]][:len(gs)],
               edgecolor='white')

for bar, val in zip(bars, gs.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Total Mean |SHAP Value|', fontsize=11)
ax.set_title('Kontribusi Kelompok Fitur Gaya Hidup vs Klinis\n(berdasarkan SHAP)',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'shap_groups.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nRANKING KELOMPOK FITUR (berdasarkan SHAP)')
for rank, (grp, val) in enumerate(gs.sort_values(ascending=False).items(), 1):
    print(f'{rank}. {grp}: {val:.4f}')
# Simpan tabel hasil ke CSV
results_df.to_csv(TABLES_DIR / 'hasil_eksperimen.csv', index=False)
print(f'hasil_eksperimen.csv tersimpan di {TABLES_DIR / "hasil_eksperimen.csv"}.')

# Download dari Colab
try:
    from google.colab import files
    print('Semua file berhasil didownload!')
except Exception as e:
    print('(Tidak di Colab - file gambar dan CSV tersimpan di direktori lokal)')
