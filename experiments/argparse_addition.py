# Add this import at the top of automated_experiments.py
import argparse

# Replace the run_experiments() function with this version:
def run_experiments(datasets=['val', 'test']):
    """Run automated experiments for specified datasets"""
    
    # Define hyperparameter variations
    base_params = {'creg': 1.0, 'sreg': 0.3, 'sizereg': 1.0}
    
    # Define variations for each parameter
    param_variations = {
        'creg': [0.5, 1.5],      # w^c variations
        'sreg': [0.1, 0.6, 1.2],      # w^s variations  
        'sizereg': [0.2, 0.5, 0.8]    # mask-area adaptive adjustment variations
    }
    
    # Process validation images if requested
    if 'val' in datasets:
        print("=== Processing Validation Dataset ===")
        val_indices = list(range(len(val_prompt)))
        for img_idx in val_indices:
            print(f"\n--- Processing Validation Image {img_idx} ---")
            process_single_image(img_idx, 'val', base_params, param_variations)
    
    # Process test images if requested
    if 'test' in datasets:
        print("\n=== Processing Test Dataset ===")
        test_indices = list(range(len(test_prompt)))
        for img_idx in test_indices:
            print(f"\n--- Processing Test Image {img_idx} ---")
            process_single_image(img_idx, 'test', base_params, param_variations)

# Replace the main section with this:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DenseDiffusion automated experiments')
    parser.add_argument('--datasets', nargs='+', choices=['val', 'test', 'both'], 
                       default=['both'], 
                       help='Which datasets to process: val, test, or both (default: both)')
    
    args = parser.parse_args()
    
    # Handle 'both' option
    if 'both' in args.datasets:
        datasets_to_process = ['val', 'test']
    else:
        datasets_to_process = args.datasets
    
    print("Starting DenseDiffusion automated experiments...")
    print(f"Processing datasets: {', '.join(datasets_to_process)}")
    
    run_experiments(datasets_to_process)
    
    print("\nExperiments completed!")
