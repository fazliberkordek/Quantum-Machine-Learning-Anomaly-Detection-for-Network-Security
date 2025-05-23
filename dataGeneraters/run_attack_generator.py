from attack_random_generator import generate_random_dataset
import pandas as pd
import os

def run_generator(n_samples=100, seed=42):
    """
    Run the attack random generator and return the generated dataset
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Generated dataset
    """
    print(f"Generating {n_samples} random samples...")
    new_data = generate_random_dataset(n_samples, seed)
    
    # Save the dataset
    output_file = "CICIDS2017_sample.csv"
    new_data.to_csv(output_file, index=False)
    print(f"\nDataset has been saved to: {os.path.abspath(output_file)}")
    print("\nDataset Preview:")
    print(new_data.head())
    print("\nLabel Distribution:")
    print(new_data['Label'].value_counts())
    
    return new_data

if __name__ == "__main__":
    # Example usage
    df = run_generator(n_samples=100) 