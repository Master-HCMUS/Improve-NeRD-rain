# Adaptive NeRD-Rain Training Script

## Setup and Running Instructions

### Option 1: Using Conda Environment (Recommended)
```bash
# Activate the data-formulator environment
conda activate data-formulator

# Navigate to the Adaptive directory
cd "path/to/Improve-NeRD-Rain/Adaptive"

# Run the training script
python train_adaptive_simple.py --batch_size 2 --num_epochs 1
```

### Option 2: Using Python directly
```bash
# If python is not in PATH, try:
python3 train_adaptive_simple.py --batch_size 2 --num_epochs 1

# Or use the full path to python
/path/to/python train_adaptive_simple.py --batch_size 2 --num_epochs 1
```

### Option 3: Using Jupyter/Colab
If running in Jupyter or Google Colab:
```python
!cd /content/Improve-NeRD-rain/Adaptive && python train_adaptive_simple.py --batch_size 2 --num_epochs 1
```

## Fixed Issues Summary

1. **Data Loading Issue**: Fixed unpacking error where data loader returns 3 values (target_imgs, input_imgs, filename) but training loop expected only 2.

2. **MLP Dimension Mismatch**: Fixed input dimension calculation in AdaptiveINR where positional encoding wasn't being included properly.

3. **TensorBoard Dependency**: Created simplified version without TensorBoard to avoid numpy/tensorflow version conflicts.

## Key Changes Made

### 1. Fixed Data Loader Unpacking
**Before:**
```python
for batch_idx, (input_imgs, target_imgs) in enumerate(progress_bar):
```

**After:**
```python
for batch_idx, (target_imgs, input_imgs, filename) in enumerate(progress_bar):
```

### 2. Fixed MLP Input Dimensions
**Before:**
```python
inp_mlp = torch.cat([q_feat, rel_coord], dim=-1)  # Missing positional encoding
```

**After:**
```python
rel_coord_enc = self.positional_encoding(rel_coord, L=L)
inp_mlp = torch.cat([q_feat, rel_coord, rel_coord_enc], dim=-1)  # Complete input
```

### 3. Removed TensorBoard Dependencies
- Removed `from torch.utils.tensorboard import SummaryWriter`
- Removed writer parameter from training functions
- Added direct logging to console instead

## Expected Dimension Calculation
For `dim=48`, `L=4`, `feat_unfold=True`, `cell_decode=True`:
- Feature dimension: `48 * 9 = 432` (after unfolding)
- Relative coordinates: `2`
- Positional encoding: `4 * L = 16`
- Cell decoding: `2`
- **Total: `432 + 2 + 16 + 2 = 452`** ✓

## Test Command
```bash
python train_adaptive_simple.py --batch_size 2 --num_epochs 1 --data_path ./Datasets
```

This will run a quick test with:
- Small batch size (2) for faster testing
- Single epoch to verify everything works
- Default dataset path

## Full Training Command
```bash
python train_adaptive_simple.py --batch_size 8 --num_epochs 500 --data_path ./Datasets --save_every 25
```