import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from model import Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_batch_norm(model):
    has_bn = False
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            has_bn = True
            break
    return has_bn

def check_dropout(model):
    has_dropout = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            has_dropout = True
            break
    return has_dropout

def check_gap_vs_fc(model):
    has_gap = False
    has_fc = False
    for module in model.modules():
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            has_gap = True
        if isinstance(module, torch.nn.Linear):
            has_fc = True
    return has_gap, has_fc

def main():
    model = Net()
    results = []
    checks_passed = True

    # Check 1: Parameter Count
    param_count = count_parameters(model)
    results.append(f"Total Parameters: {param_count:,}")
    if param_count > 20000:
        results.append("❌ Parameter count exceeds 20,000 limit")
        checks_passed = False
    else:
        results.append("✅ Parameter count within limit")

    # Check 2: Batch Normalization
    if check_batch_norm(model):
        results.append("✅ Model uses Batch Normalization")
    else:
        results.append("❌ Model does not use Batch Normalization")
        checks_passed = False

    # Check 3: Dropout
    if check_dropout(model):
        results.append("✅ Model uses Dropout")
    else:
        results.append("❌ Model does not use Dropout")
        checks_passed = False

    # Check 4: GAP vs FC
    has_gap, has_fc = check_gap_vs_fc(model)
    if has_gap and not has_fc:
        results.append("✅ Model uses Global Average Pooling (preferred)")
    elif has_fc:
        results.append("⚠️ Model uses Fully Connected layer (consider using GAP)")
    else:
        results.append("❌ Model uses neither GAP nor FC layer")
        checks_passed = False

    # Write results to file
    with open('model_check_results.txt', 'w') as f:
        f.write('\n'.join(results))

    # Exit with appropriate status
    sys.exit(0 if checks_passed else 1)

if __name__ == "__main__":
    main() 