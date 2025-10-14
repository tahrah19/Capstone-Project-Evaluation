"""
Automated Model Comparison Script
Statistical Analysis + Precision-Recall Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AUTOMATED MODEL COMPARISON TOOL")
print("Statistical Analysis + Precision-Recall Trade-off")
print("="*80)

# ==============================================================================
# STEP 1: AUTO-LOAD ALL BERTSCORE FILES
# ==============================================================================

print("\n[1/4] Loading BERTScore files...")

# Get all CSV and Excel files
all_files = glob.glob("*.csv") + glob.glob("*.xlsx") + glob.glob("*.xls")

# Filter for files containing "bert" and "score" (case-insensitive)
files = []
for file in all_files:
    file_lower = file.lower()
    if 'bert' in file_lower and 'score' in file_lower:
        files.append(file)

if not files:
    print("❌ ERROR: No BERTScore files found!")
    print("Make sure files contain 'bert' and 'score' in filename")
    exit()

csv_files = [f for f in files if f.lower().endswith('.csv')]
xlsx_files = [f for f in files if f.lower().endswith(('.xlsx', '.xls'))]

print(f"Found {len(files)} files ({len(csv_files)} CSV, {len(xlsx_files)} Excel):")

all_data = []

for file in files:
    print(f"  📄 {file}")
    
    # Detect model name (case-insensitive)
    filename_lower = file.lower()
    if 'llama3.2' in filename_lower or 'llama-3.2' in filename_lower:
        model = 'llama3.2'
    elif 'gemma3' in filename_lower or 'gemma-3' in filename_lower:
        model = 'gemma3'
    elif 'phi4-mini' in filename_lower or 'phi4mini' in filename_lower:
        model = 'phi4-mini'
    else:
        print(f"     ⚠️  Could not detect model name (looking for: llama3.2, gemma3, phi4-mini)")
        continue
    
    # Extract document name (just use the filename, we'll clean it up)
    doc_name = file.split('.')[0]  # Remove extension
    
    # Load file
    try:
        if file.lower().endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Check for required columns
        required_cols = ['BERT_PRECISION', 'BERT_RECALL', 'BERT_F1']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"     ⚠️  Missing columns: {missing_cols}")
            continue
        
        # Rename columns
        df = df.rename(columns={
            'BERT_PRECISION': 'precision',
            'BERT_RECALL': 'recall',
            'BERT_F1': 'f1'
        })
        
        # Add metadata
        df['model'] = model
        df['document'] = doc_name
        
        all_data.append(df[['model', 'document', 'precision', 'recall', 'f1']])
        
    except Exception as e:
        print(f"     ❌ Error loading: {e}")
        continue

if not all_data:
    print("\n❌ No valid data loaded.")
    exit()

# Combine all data
df_all = pd.concat(all_data, ignore_index=True)

print(f"\n✓ Successfully loaded {len(df_all)} total questions")
print(f"  Models: {df_all['model'].unique().tolist()}")
print(f"  Documents: {len(df_all['document'].unique())} documents")

# ==============================================================================
# STEP 2: STATISTICAL ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("[2/4] STATISTICAL ANALYSIS")
print("="*80)

print("\n📊 OVERALL PERFORMANCE")
print("-"*80)

overall_stats = df_all.groupby('model').agg({
    'f1': ['mean', 'std', 'min', 'max', 'count'],
    'precision': 'mean',
    'recall': 'mean'
}).round(4)

overall_stats.columns = ['Mean_F1', 'Std_F1', 'Min_F1', 'Max_F1', 'N_Questions', 'Mean_Precision', 'Mean_Recall']
overall_stats = overall_stats.sort_values('Mean_F1', ascending=False)

print(overall_stats)

winner = overall_stats.index[0]
winner_f1 = overall_stats.loc[winner, 'Mean_F1']
winner_std = overall_stats.loc[winner, 'Std_F1']

print(f"\n🏆 WINNER: {winner} (F1 = {winner_f1:.4f} ± {winner_std:.4f})")

# Consistency
print("\n📈 CONSISTENCY (Lower Std Dev = More Consistent)")
print("-"*80)
consistency = overall_stats[['Std_F1']].sort_values('Std_F1')
print(consistency)

most_consistent = consistency.index[0]
least_consistent = consistency.index[-1]
print(f"\n✓ Most consistent: {most_consistent} (std = {consistency.loc[most_consistent, 'Std_F1']:.4f})")
print(f"✗ Least consistent: {least_consistent} (std = {consistency.loc[least_consistent, 'Std_F1']:.4f})")

# Statistical Tests
print("\n🔬 STATISTICAL SIGNIFICANCE TESTS")
print("-"*80)

models = df_all['model'].unique()

if len(models) >= 2:
    print("\nPairwise T-Tests:")
    test_results = []
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model_a = models[i]
            model_b = models[j]
            
            scores_a = df_all[df_all['model'] == model_a]['f1'].values
            scores_b = df_all[df_all['model'] == model_b]['f1'].values
            
            t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
            mean_diff = scores_a.mean() - scores_b.mean()
            pooled_std = np.sqrt((scores_a.std()**2 + scores_b.std()**2) / 2)
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            if abs(cohen_d) < 0.2:
                effect = "negligible"
            elif abs(cohen_d) < 0.5:
                effect = "small"
            elif abs(cohen_d) < 0.8:
                effect = "medium"
            else:
                effect = "large"
            
            significant = "YES ✓" if p_value < 0.05 else "NO ✗"
            better_model = model_a if mean_diff > 0 else model_b
            
            print(f"\n  {model_a} vs {model_b}:")
            print(f"    Mean difference: {mean_diff:+.4f}")
            print(f"    t-statistic: {t_stat:.4f}")
            print(f"    p-value: {p_value:.4f}")
            print(f"    Cohen's d: {cohen_d:.4f} ({effect} effect)")
            print(f"    Significant (p<0.05)?: {significant}")
            if p_value < 0.05:
                print(f"    → {better_model} performs significantly better")
            
            test_results.append({
                'Comparison': f"{model_a} vs {model_b}",
                'Mean_Diff': round(mean_diff, 4),
                't_statistic': round(t_stat, 4),
                'p_value': round(p_value, 4),
                'Cohens_d': round(cohen_d, 4),
                'Effect_Size': effect,
                'Significant': significant,
                'Better_Model': better_model if p_value < 0.05 else 'No difference'
            })
    
    df_tests = pd.DataFrame(test_results)

# ==============================================================================
# STEP 3: PRECISION-RECALL ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("[3/4] PRECISION vs RECALL ANALYSIS")
print("="*80)

pr_analysis = df_all.groupby('model').agg({
    'precision': ['mean', 'std'],
    'recall': ['mean', 'std']
}).round(4)

pr_analysis.columns = ['Avg_Precision', 'Std_Precision', 'Avg_Recall', 'Std_Recall']
pr_analysis['PR_Balance'] = pr_analysis['Avg_Precision'] - pr_analysis['Avg_Recall']

def classify_strategy(balance):
    if balance > 0.03:
        return "Precision-focused"
    elif balance < -0.03:
        return "Recall-focused"
    else:
        return "Balanced"

pr_analysis['Strategy'] = pr_analysis['PR_Balance'].apply(classify_strategy)
pr_analysis = pr_analysis.sort_values('Avg_Precision', ascending=False)

print("\n📊 PRECISION vs RECALL COMPARISON")
print("-"*80)
print(pr_analysis)

highest_precision = pr_analysis.index[0]
highest_recall = pr_analysis.sort_values('Avg_Recall', ascending=False).index[0]

print(f"\n🎯 INSIGHTS:")
print(f"  • Highest Precision: {highest_precision} ({pr_analysis.loc[highest_precision, 'Avg_Precision']:.4f})")
print(f"    → Best for legal use (minimizes hallucinations)")
print(f"  • Highest Recall: {highest_recall} ({pr_analysis.loc[highest_recall, 'Avg_Recall']:.4f})")
print(f"    → Captures most information (but may include errors)")

print("\n💡 STRATEGY INTERPRETATION:")
for model in pr_analysis.index:
    strategy = pr_analysis.loc[model, 'Strategy']
    p = pr_analysis.loc[model, 'Avg_Precision']
    r = pr_analysis.loc[model, 'Avg_Recall']
    
    if strategy == "Precision-focused":
        print(f"  • {model}: {strategy}")
        print(f"    → Prioritizes accuracy over completeness (P={p:.3f} > R={r:.3f})")
        print(f"    → Safer for legal contexts (fewer hallucinations)")
    elif strategy == "Recall-focused":
        print(f"  • {model}: {strategy}")
        print(f"    → Prioritizes completeness over accuracy (R={r:.3f} > P={p:.3f})")
        print(f"    → More verbose but less reliable")
    else:
        print(f"  • {model}: {strategy}")
        print(f"    → Equal emphasis on precision and recall (P≈R)")

# ==============================================================================
# STEP 4: SAVE RESULTS
# ==============================================================================

print("\n" + "="*80)
print("[4/4] SAVING RESULTS")
print("="*80)

with pd.ExcelWriter('comparison_results.xlsx', engine='openpyxl') as writer:
    overall_stats.to_excel(writer, sheet_name='Overall_Performance')
    pr_analysis.to_excel(writer, sheet_name='Precision_Recall')
    if len(models) >= 2:
        df_tests.to_excel(writer, sheet_name='Statistical_Tests', index=False)
    df_all.to_excel(writer, sheet_name='All_Data', index=False)

print("✓ Saved: comparison_results.xlsx")

# ==============================================================================
# GENERATE REPORT TEXT
# ==============================================================================

print("\n" + "="*80)
print("📝 REPORT TEXT FOR YOUR DISCUSSION SECTION")
print("="*80)

print("\n=== COPY THIS INTO YOUR DISCUSSION ===\n")
print("**Comparison Between Model Outputs:**\n")

second = overall_stats.index[1] if len(overall_stats) > 1 else None
third = overall_stats.index[2] if len(overall_stats) > 2 else None
second_f1 = overall_stats.loc[second, 'Mean_F1'] if second else 0
third_f1 = overall_stats.loc[third, 'Mean_F1'] if third else 0
improvement = ((winner_f1 - second_f1) / second_f1 * 100) if second else 0

print(f"Our evaluation of three local LLMs using BERTScore metrics revealed {winner} as the")
print(f"superior performer, achieving an average F1-score of {winner_f1:.4f} (±{winner_std:.4f}),")
print(f"outperforming {second} ({second_f1:.4f}) and {third} ({third_f1:.4f}). This represents")
print(f"a {improvement:.1f}% improvement over the second-best model.")

if len(models) >= 2 and 'df_tests' in locals():
    sig_comparisons = df_tests[df_tests['Significant'] == 'YES ✓']
    if not sig_comparisons.empty:
        print(f"\nStatistical testing confirmed these differences were significant. Specifically,")
        for _, row in sig_comparisons.iterrows():
            print(f"{row['Better_Model']} significantly outperformed its competitor (p={row['p_value']:.3f},")
            print(f"Cohen's d={row['Cohens_d']:.3f}, {row['Effect_Size']} effect size).")

print(f"\n**Precision vs Recall Trade-offs:**\n")
print(f"Analysis of precision and recall components revealed distinct model strategies.")
print(f"{highest_precision} demonstrated a {pr_analysis.loc[highest_precision, 'Strategy'].lower()} approach")
print(f"(Precision={pr_analysis.loc[highest_precision, 'Avg_Precision']:.3f}, Recall={pr_analysis.loc[highest_precision, 'Avg_Recall']:.3f}),")
print(f"prioritizing accuracy over completeness—a critical attribute for legal document")
print(f"analysis where hallucinations carry serious consequences.")

recall_focused = pr_analysis[pr_analysis['Strategy'] == 'Recall-focused']
if not recall_focused.empty:
    rf_model = recall_focused.index[0]
    print(f"\n{rf_model} exhibited {pr_analysis.loc[rf_model, 'Strategy'].lower()} behavior")
    print(f"(Precision={pr_analysis.loc[rf_model, 'Avg_Precision']:.3f}, Recall={pr_analysis.loc[rf_model, 'Avg_Recall']:.3f}),")
    print(f"capturing more content but with reduced accuracy.")

balanced = pr_analysis[pr_analysis['Strategy'] == 'Balanced']
if not balanced.empty:
    bal_model = balanced.index[0]
    print(f"\n{bal_model} maintained the most balanced precision-recall profile")
    print(f"(Precision={pr_analysis.loc[bal_model, 'Avg_Precision']:.3f}, Recall={pr_analysis.loc[bal_model, 'Avg_Recall']:.3f}).")

print(f"\n**Model Consistency:**\n")
print(f"{most_consistent} demonstrated the highest consistency (std={consistency.loc[most_consistent, 'Std_F1']:.4f}),")
print(f"indicating reliable performance across diverse questions. Conversely, {least_consistent}")
print(f"showed greater variability (std={consistency.loc[least_consistent, 'Std_F1']:.4f}), suggesting")
print(f"unpredictable performance that could be problematic in production deployment.")

print(f"\n**Implications for Legal Context:**\n")
print(f"For the Coroner Project, {highest_precision}'s high-precision strategy makes it most")
print(f"appropriate for deployment in legal settings where factual accuracy is paramount.")
print(f"Incomplete answers (which can be flagged for human review) are preferable to")
print(f"hallucinated information (which may go undetected and mislead case analysis).")

print("\n=== END OF REPORT TEXT ===")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated:")
print("  1. comparison_results.xlsx - All tables")
print("  2. Report text above - Copy into Discussion")
print("="*80)