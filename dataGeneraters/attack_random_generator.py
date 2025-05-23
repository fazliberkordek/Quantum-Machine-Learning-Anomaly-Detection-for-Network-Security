import numpy as np
import pandas as pd
import os

def generate_random_dataset(n_samples, seed=42):
    """
    Generate random network traffic data based on CICIDS2017 format
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Generate random data with reasonable ranges
    new_data = pd.DataFrame({
        "Flow Duration": np.random.randint(50000, 1000000, size=n_samples),
        "Total Fwd Packets": np.random.randint(1, 100, size=n_samples),
        "Total Backward Packets": np.random.randint(1, 100, size=n_samples),
        "Flow Bytes/s": np.random.randint(100, 10000, size=n_samples),
        "Flow Packets/s": np.round(np.random.uniform(0.01, 2.0, size=n_samples), 2),
        "Label": np.random.choice(["BENIGN", "DoS", "PortScan", "BruteForce"], size=n_samples)
    })
    
    return new_data

def main():
    # Read the original dataset to understand its structure
    try:
        original_df = pd.read_csv("CICIDS2017_sample.csv")
        print("Successfully loaded CICIDS2017_sample.csv")
        print("\nDataset Info:")
        print(f"Number of samples: {len(original_df)}")
        print(f"Columns: {', '.join(original_df.columns)}")
        print(f"Unique labels: {original_df['Label'].unique()}")
    except FileNotFoundError:
        print("Warning: CICIDS2017_sample.csv not found. Proceeding with default values.")
    
    # Get user input
    while True:
        try:
            n_samples = int(input("\nHow many samples would you like to generate? (default: 100): ") or "100")
            if n_samples <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Generate the dataset
    print(f"\nGenerating {n_samples} random samples...")
    new_data = generate_random_dataset(n_samples)
    
    # Save the dataset
    output_file = "generated_network_traffic.csv"
    new_data.to_csv(output_file, index=False)
    print(f"\nDataset has been saved to: {os.path.abspath(output_file)}")
    print("\nDataset Preview:")
    print(new_data.head())
    print("\nLabel Distribution:")
    print(new_data['Label'].value_counts())

if __name__ == "__main__":
    main()