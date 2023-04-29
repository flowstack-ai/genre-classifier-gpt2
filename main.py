import torch
from GPTGC import GPTGC

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-> Using device: ", device)

model = GPTGC(device, fine_tune=True)
model.fit()