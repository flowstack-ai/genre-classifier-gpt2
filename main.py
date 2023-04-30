import torch
from GPTGC import GPTGC

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-> Using device: ", device)

# Set fine-tune = True to fine-tune the model.
# model = GPTGC(device, fine_tune=True) # Loading model for training.
# model.fit() # Training the model.

# Set fine-tune = False or remove the parameter to load the model for inference.
model = GPTGC(device) # Loading model for inference.

# Perform inference on a single text.
model.predict("Your movie description here.")