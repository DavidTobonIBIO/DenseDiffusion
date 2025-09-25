# DenseDiffusion Automated Experiments

This script performs automated experiments on the DenseDiffusion model to analyze the effects of different hyperparameters on image generation quality.

## Overview

The experiment generates images using validation dataset samples (images 1.png and 5.png) with different hyperparameter configurations:

- **Base generation**: Uses default hyperparameters from the Gradio app
- **Parameter variations**: Tests 2 different values for each of the 3 key hyperparameters
- **Total generations**: 8 images per input (1 base + 7 variations)

## Hyperparameters Tested

1. **$w^c$ (creg)**: Cross-attention modulation strength
   - Default: 1.0
   - Variations: 0.5, 1.5

2. **$w^s$ (sreg)**: Self-attention modulation strength  
   - Default: 0.3
   - Variations: 0.1, 0.6, 1.2

3. **sizereg**: Mask-area adaptive adjustment degree
   - Default: 1.0
   - Variations: 0.5, 0.8

## Setup

*Start from the root directory of the DenseDiffusion repository*

1. **Create and activate conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate dense_diffusion_env
   ```

2. **Environment setup**:
   - Create a `.env` file with your Hugging Face token:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```
   - Ensure you have CUDA available for GPU acceleration

3. **Dataset**:
   - Ensure the validation dataset is available at `./dataset/valset.pkl`
   - Ensure layout images are available at `./dataset/valset_layout/`

## Usage

Run the automated experiments:

```bash
conda activate dense_diffusion_env
cd experiments
python automated_experiments.py
```

The script will:
1. Load validation samples for images 1 and 5
2. Generate images with base hyperparameters
3. Generate variations for each hyperparameter
4. Create matplotlib visualizations showing all results
5. Save the visualizations as PNG files with timestamps

## Output

- **Console output**: Progress updates and parameter information
- **Image files**: High-resolution PNG files with subplot grids
  - Format: `densediffusion_experiments_image_{idx}_{timestamp}.png`
  - Each subplot shows the generated image with parameter values in the title

## Expected Results

The experiment will produce:
- 2 visualization files (one for each input image)
- Each visualization contains 7 subplots arranged in a 3x3 grid
- Clear parameter labels showing the effect of each hyperparameter variation

## Notes

- **GPU Memory**: Ensure sufficient VRAM (recommended: 8GB+)
- **Runtime**: Each image generation takes ~20-60 seconds depending on hardware
- **Reproducibility**: Uses fixed seeds from the validation dataset for consistent results
- **Environment**: Uses the existing `dense_diffusion_env` conda environment with compatible package versions

<!-- ## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or use CPU (slower)
2. **Environment issues**: Ensure conda environment is activated
3. **Authentication errors**: Verify HF_TOKEN in .env file
4. **File not found**: Ensure dataset files are properly extracted -->

## File Structure

```
DenseDiffusion/
├── dataset/
│   |── automated_experiments.py # Main experiment script
|   └── README_experiments.md      # This file
├── utils.py
├── gradio_app.py             # Original Gradio application
├── environment.yml           # Conda environment specification
├── dataset/
│   ├── valset.pkl            # Validation prompts
│   └── valset_layout/        # Layout images
│       ├── 1.png
│       ├── 5.png
│       └── ...
└── .env                      # Environment variables
```
