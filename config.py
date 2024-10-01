import sys
# -----------------
# DATASET ROOTS
# -----------------
sys.path.append('/local_datasets')
cifar_10_root = '/local_datasets/khpark/AND'
cifar_100_root = '/local_datasets/khpark/AND'
cub_root = '/local_datasets/khpark/AND'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = 'MetaGCD/pretrained_weight/dino_vitbase16_pretrain.pth'
# dino_head_path = 
feature_extract_dir = '/MetaGCD/extracted_features_public_impl'     # Extract features to this directory
exp_root = '/data/pgh2874/Anytime_Novel_Discovery/MetaGCD/'          # All logs and checkpoints will be saved here