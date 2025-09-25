import numpy as np
import torch
import pickle
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image

import diffusers
from diffusers import DDIMScheduler
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import preprocess_mask, process_example

# Check diffusers version
if diffusers.__version__ != '0.20.2':
    print("Please use diffusers v0.20.2")
    sys.exit(0)

# Global variables (same as in gradio_app.py)
global sreg, creg, sizereg, COUNT, creg_maps, sreg_maps, pipe, text_cond

sreg = 0
creg = 0
sizereg = 0
COUNT = 0
reg_sizes = {}
creg_maps = {}
sreg_maps = {}
text_cond = 0
device = "cuda"

# Load environment and model
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

pipe = diffusers.StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    variant="fp16",
    cache_dir='../models/diffusers/',
    use_auth_token=HF_TOKEN).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)
timesteps = pipe.scheduler.timesteps
sp_sz = pipe.unet.sample_size

# Load validation and test datasets
with open('../dataset/valset.pkl', 'rb') as f:
    val_prompt = pickle.load(f)
val_layout = '../dataset/valset_layout/'

with open('../dataset/testset.pkl', 'rb') as f:
    test_prompt = pickle.load(f)
test_layout = '../dataset/testset_layout/'

# Modified forward function (same as in gradio_app.py)
def mod_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
    residual = hidden_states

    if self.spatial_norm is not None:
        hidden_states = self.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states)

    global sreg, creg, COUNT, creg_maps, sreg_maps, reg_sizes, text_cond
    
    sa_ = True if encoder_hidden_states is None else False
    encoder_hidden_states = text_cond if encoder_hidden_states is not None else hidden_states
        
    if self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = self.head_to_batch_dim(query)
    key = self.head_to_batch_dim(key)
    value = self.head_to_batch_dim(value)
    
    if COUNT/32 < 50*0.3:
        
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()
            
        sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                                        dtype=query.dtype, device=query.device),
                            query, key.transpose(-1, -2), beta=0, alpha=self.scale)
        
        treg = torch.pow(timesteps[COUNT//32]/1000, 5)
        
        ## reg at self-attn
        if sa_:
            min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
            mask = sreg_maps[sim.size(1)].repeat(self.heads,1,1)
            size_reg = reg_sizes[sim.size(1)].repeat(self.heads,1,1)
            
            sim[int(sim.size(0)/2):] += (mask>0)*size_reg*sreg*treg*(max_value-sim[int(sim.size(0)/2):])
            sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*sreg*treg*(sim[int(sim.size(0)/2):]-min_value)
            
        ## reg at cross-attn
        else:
            min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
            mask = creg_maps[sim.size(1)].repeat(self.heads,1,1)
            size_reg = reg_sizes[sim.size(1)].repeat(self.heads,1,1)
            
            sim[int(sim.size(0)/2):] += (mask>0)*size_reg*creg*treg*(max_value-sim[int(sim.size(0)/2):])
            sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*creg*treg*(sim[int(sim.size(0)/2):]-min_value)

        attention_probs = sim.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)
            
    else:
        attention_probs = self.get_attention_scores(query, key, attention_mask)
           
    COUNT += 1
            
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if self.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / self.rescale_output_factor

    return hidden_states

# Apply modified forward to all attention modules
for _module in pipe.unet.modules():
    if _module.__class__.__name__ == "Attention":
        _module.__class__.__call__ = mod_forward

def process_generation(binary_matrixes, seed, creg_, sreg_, sizereg_, bsz, master_prompt, *prompts):
    """Modified version of process_generation from gradio_app.py"""
    global creg, sreg, sizereg
    creg, sreg, sizereg = creg_, sreg_, sizereg_
    
    clipped_prompts = prompts[:len(binary_matrixes)]
    prompts = [master_prompt] + list(clipped_prompts)
    layouts = torch.cat([preprocess_mask(mask_, sp_sz, sp_sz, device) for mask_ in binary_matrixes])
    
    text_input = pipe.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    cond_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    uncond_input = pipe.tokenizer([""]*bsz, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    
    ###########################
    ###### prep for sreg ###### 
    ###########################
    global sreg_maps, reg_sizes
    sreg_maps = {}
    reg_sizes = {}
    
    for r in range(4):
        res = int(sp_sz/np.power(2,r))
        layouts_s = F.interpolate(layouts,(res, res),mode='nearest')
        layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1)
        reg_sizes[np.power(res, 2)] = 1-sizereg*layouts_s.sum(-1, keepdim=True)/(np.power(res, 2))
        sreg_maps[np.power(res, 2)] = layouts_s


    ###########################
    ###### prep for creg ######
    ###########################
    pww_maps = torch.zeros(1,77,sp_sz,sp_sz).to(device)
    for i in range(1,len(prompts)):
        wlen = text_input['length'][i] - 2
        widx = text_input['input_ids'][i][1:1+wlen]
        for j in range(77):
            try:
                if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    pww_maps[:,j:j+wlen,:,:] = layouts[i-1:i]
                    cond_embeddings[0][j:j+wlen] = cond_embeddings[i][1:1+wlen]
                    break
            except:
                print("Error: Please check whether every segment prompt is included in the full text!")
                return None
    
    global creg_maps
    creg_maps = {}
    for r in range(4):
        res = int(sp_sz/np.power(2,r))
        layout_c = F.interpolate(pww_maps,(res,res),mode='nearest').view(1,77,-1).permute(0,2,1).repeat(bsz,1,1)
        creg_maps[np.power(res, 2)] = layout_c


    ###########################    
    #### prep for text_emb ####
    ###########################
    global text_cond
    text_cond = torch.cat([uncond_embeddings, cond_embeddings[:1].repeat(bsz,1,1)])    
    
    global COUNT
    COUNT = 0
    
    if seed == -1:
        latents = torch.randn(bsz,4,sp_sz,sp_sz).to(device)
    else:
        latents = torch.randn(bsz,4,sp_sz,sp_sz, generator=torch.Generator().manual_seed(seed)).to(device)
        
    image = pipe(prompts[:1]*bsz, latents=latents).images

    return image

def load_example_data(image_idx, dataset_type='val'):
    """Load example data for a given image index from validation or test set"""
    if dataset_type == 'val':
        layout_path = val_layout + f'{image_idx}.png'
        prompt_data = val_prompt[image_idx]
        # Use seeds from gradio_app.py examples
        seeds = {0: 381940206, 1: 307504592, 5: 114972190}
        seed = seeds.get(image_idx, 42 + image_idx)
    else:  # test
        layout_path = test_layout + f'{image_idx}.png'
        prompt_data = test_prompt[image_idx]
        seed = 42 + image_idx  # Generate consistent seeds for test set
    
    all_prompts = '***'.join([prompt_data['textual_condition']] + prompt_data['segment_descriptions'])
    
    # Process the example to get binary matrices and prompts
    result = process_example(layout_path, all_prompts, seed)
    
    # Extract the relevant data
    binary_matrixes = result[1]
    prompts = result[2+24:2+24+12]  # Extract prompts from the result
    general_prompt = result[-2].value if hasattr(result[-2], 'value') else prompt_data['textual_condition']
    
    return binary_matrixes, prompts, general_prompt, seed, layout_path

def run_experiments():
    """Run automated experiments for all validation and test images"""
    
    # Define hyperparameter variations
    base_params = {'creg': 1.0, 'sreg': 0.3, 'sizereg': 1.0}
    
    # Define variations for each parameter
    param_variations = {
        'creg': [0.5, 1.5],      # w^c variations
        'sreg': [0.1, 0.6, 1.2],      # w^s variations  
        'sizereg': [0.2, 0.5, 0.8]    # mask-area adaptive adjustment variations
    }
    
    # Process all validation images
    print("=== Processing Validation Dataset ===")
    val_indices = list(range(len(val_prompt)))
    for img_idx in val_indices:
        print(f"\n--- Processing Validation Image {img_idx} ---")
        process_single_image(img_idx, 'val', base_params, param_variations)
    
    # Process all test images
    print("\n=== Processing Test Dataset ===")
    test_indices = list(range(len(test_prompt)))
    for img_idx in test_indices:
        print(f"\n--- Processing Test Image {img_idx} ---")
        process_single_image(img_idx, 'test', base_params, param_variations)

def process_single_image(img_idx, dataset_type, base_params, param_variations):
    """Process a single image with all parameter variations"""
    try:
        # Load example data
        binary_matrixes, prompts, general_prompt, seed, layout_path = load_example_data(img_idx, dataset_type)
        
        # Filter out empty prompts
        valid_prompts = [p.value if hasattr(p, 'value') else str(p) for p in prompts if p is not None]
        valid_prompts = [p for p in valid_prompts if p.strip()]
        
        print(f"General prompt: {general_prompt}")
        print(f"Segment prompts: {valid_prompts}")
        
        # Store all generated images and their parameters
        all_images = []
        all_param_info = []
        
        # Add layout image as first image
        if os.path.exists(layout_path):
            layout_img = Image.open(layout_path)
            all_images.append(layout_img)
            all_param_info.append({
                'params': {},
                'changed_param': None,
                'is_layout': True
            })
        
        # 1. Generate with base parameters
        print("Generating with base parameters...")
        base_images = process_generation(
            binary_matrixes, seed, 
            base_params['creg'], base_params['sreg'], base_params['sizereg'], 
            1, general_prompt, *valid_prompts
        )
        
        if base_images:
            all_images.append(base_images[0])
            all_param_info.append({
                'params': base_params.copy(),
                'changed_param': None,
                'is_base': True
            })
        
        # 2. Generate variations for each parameter
        for param_name, variations in param_variations.items():
            for variation in variations:
                print(f"Generating with {param_name}={variation}...")
                
                # Create parameter set with one variation
                params = base_params.copy()
                params[param_name] = variation
                
                var_images = process_generation(
                    binary_matrixes, seed,
                    params['creg'], params['sreg'], params['sizereg'],
                    1, general_prompt, *valid_prompts
                )
                
                if var_images:
                    all_images.append(var_images[0])
                    all_param_info.append({
                        'params': params,
                        'changed_param': param_name,
                        'is_base': False
                    })
        
        # 3. Create simple visualization
        if all_images:
            create_simple_visualization(
                all_images, all_param_info, img_idx, dataset_type,
                general_prompt, base_params
            )
        
        print(f"Completed experiments for {dataset_type} image {img_idx}")
        
    except Exception as e:
        print(f"Error processing {dataset_type} image {img_idx}: {str(e)}")
        print("Continuing with next image...")

def create_simple_visualization(images, param_info, img_idx, dataset_type, 
                              general_prompt, base_params):
    """Create simple matplotlib visualization with proper bold formatting"""
    
    n_images = len(images)
    cols = 4
    rows = (n_images + cols - 1) // cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    
    # Main title with prompt
    fig.suptitle(f'{dataset_type.capitalize()} Image {img_idx}: {general_prompt}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Flatten axes for easier indexing
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Display images
    for i, (img, info) in enumerate(zip(images, param_info)):
        ax = axes[i]
        ax.imshow(img)
        
        # Format title with proper bold formatting
        if info.get('is_layout', False):
            ax.set_title("Input Layout", fontsize=12, fontweight='bold')
        elif info.get('is_base', False):
            title = f"BASE: wc={info['params']['creg']}, ws={info['params']['sreg']}, sizereg={info['params']['sizereg']}"
            ax.set_title(title, fontsize=10, fontweight='bold')
        else:
            # Create title with bold formatting for changed parameter
            param_names = {'creg': 'wc', 'sreg': 'ws', 'sizereg': 'sizereg'}
            changed_param = info['changed_param']
            
            title_parts = []
            for param, value in info['params'].items():
                param_display = param_names.get(param, param)
                title_parts.append(f"{param_display}={value}")
            
            title = ", ".join(title_parts)
            
            # Use different colors for different changed parameters
            colors = {'creg': 'red', 'sreg': 'blue', 'sizereg': 'green'}
            color = colors.get(changed_param, 'black')
            
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/{dataset_type}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'image_{img_idx}_{timestamp}.png'
    filepath = os.path.join(results_dir, dataset_type, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {filepath}")
    
    plt.close()

if __name__ == "__main__":
    print("Starting DenseDiffusion automated experiments...")
    print("This will generate images with different hyperparameter settings for all validation and test images")
    
    run_experiments()
    
    print("\nExperiments completed!")
