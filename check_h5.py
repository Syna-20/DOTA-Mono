import h5py
import sys

def print_structure(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"  Shape: {obj.shape}")
        print(f"  Type: {obj.dtype}")

filename = './data/training/train_part1.h5'
print(f"\nExamining {filename}:")
with h5py.File(filename, 'r') as f:
    f.visititems(print_structure) 