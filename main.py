import torch
from GPTRC import GPTRC

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-> Using device: ", device)

model = GPTRC(device, fine_tune=True)
model.fit()