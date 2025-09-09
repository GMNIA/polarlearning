"""
California Housing Dataset - Expected Performance Benchmarks

The California Housing dataset (20,640 samples, 8 features) predicts median house values.

Typical performance benchmarks:
- Linear Regression: MAE ~0.5-0.7 (on normalized targets)
- Random Forest: MAE ~0.3-0.5  
- Gradient Boosting: MAE ~0.3-0.4
- Neural Networks: MAE ~0.3-0.6 (depends on architecture/hyperparams)

Target statistics:
- Raw target range: ~$15k - $500k+ (median house values)
- After normalization: typically 0-1 or standardized (-2 to +3)

For our MLP (8→64→32→1):
- Good MAE: < 0.5 (on normalized targets)
- Acceptable MAE: 0.5-0.8
- Poor MAE: > 0.8

Hyperparameter recommendations:
- Learning rates: 0.001-0.01 (start with 0.001-0.005)
- Epochs: 100-300 with early stopping
- Architecture: Current 8→64→32→1 is reasonable
- Optimizer: SGD or Adam (Adam often better for tabular data)
"""

def analyze_mae_performance(mae_value, target_std):
    """
    Analyze if MAE is good relative to target distribution
    """
    # MAE as percentage of target standard deviation
    mae_pct = (mae_value / target_std) * 100
    
    print(f"\n📈 MAE Performance Analysis:")
    print(f"   MAE: {mae_value:.6f}")
    print(f"   Target Std Dev: {target_std:.6f}")
    print(f"   MAE as % of target std: {mae_pct:.1f}%")
    
    if mae_pct < 30:
        print("   ✅ Excellent performance (MAE < 30% of target std)")
    elif mae_pct < 50:
        print("   ✅ Good performance (MAE < 50% of target std)")
    elif mae_pct < 70:
        print("   ⚠️  Acceptable performance (MAE < 70% of target std)")
    else:
        print("   ❌ Poor performance (MAE > 70% of target std)")
        print("   💡 Suggestions:")
        print("      - Try lower learning rate (0.001 → 0.0005)")
        print("      - Try Adam optimizer instead of SGD")
        print("      - Increase model capacity (64→128, 32→64)")
        print("      - More epochs with patience")
