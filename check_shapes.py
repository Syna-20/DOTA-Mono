import h5py
import numpy as np

def print_shapes(filename):
    print(f"\nChecking shapes in {filename}:")
    try:
        with h5py.File(filename, 'r') as f:
            for key in f.keys():
                data = f[key]
                print(f"{key}: shape = {data.shape}, dtype = {data.dtype}")
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")

# Check training file
train_file = './data/training/train.h5'
print_shapes(train_file)

# Check test file
test_file = './data/test/test.h5'
print_shapes(test_file) 