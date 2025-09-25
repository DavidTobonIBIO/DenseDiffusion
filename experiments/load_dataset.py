import pickle
import numpy as np
from PIL import Image

# Load validation dataset
with open('../dataset/valset.pkl', 'rb') as f:
    val_prompt = pickle.load(f)

print("Validation dataset structure:")
print("Keys:", val_prompt.keys())
print(f"Number of samples: {len(val_prompt)}")
print("Sample keys:", val_prompt[1].keys())

print("\nSample 1 (index 1):")
print(f"Textual condition: {val_prompt[1]['textual_condition']}")
print(f"Segment descriptions: {val_prompt[1]['segment_descriptions']}")

print("\nSample 5 (index 5):")
print(f"Textual condition: {val_prompt[5]['textual_condition']}")
print(f"Segment descriptions: {val_prompt[5]['segment_descriptions']}")

# Check layout images
layout_1 = Image.open('../dataset/valset_layout/1.png')
layout_5 = Image.open('../dataset/valset_layout/5.png')

print(f"\nLayout 1 size: {layout_1.size}")
print(f"Layout 5 size: {layout_5.size}")

# Load test dataset
with open('../dataset/testset.pkl', 'rb') as f:
    test_prompt = pickle.load(f)

print("Test dataset structure:")
print("Keys:", test_prompt.keys())
print(f"Number of samples: {len(test_prompt)}")
print("Sample keys:", test_prompt[1].keys())

print("\nSample 1 (index 1):")
print(f"Textual condition: {test_prompt[1]['textual_condition']}")
print(f"Segment descriptions: {test_prompt[1]['segment_descriptions']}")

print("\nSample 5 (index 5):")
print(f"Textual condition: {test_prompt[5]['textual_condition']}")
print(f"Segment descriptions: {test_prompt[5]['segment_descriptions']}")

