import h5py

def inspect_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        print(f"\nInspecting file: {filepath}")
        print("\nDatasets:")
        for k in f.keys():
            print(f"{k}:")
            print(f"  Shape: {f[k].shape}")
            print(f"  Type:  {f[k].dtype}")
            print(f"  Size:  {f[k].size:,} elements")
            print()

if __name__ == "__main__":
    inspect_h5('./data/training/train.h5') 