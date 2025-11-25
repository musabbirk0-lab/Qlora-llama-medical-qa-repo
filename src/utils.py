import torch




def freeze_model(model):
for param in model.parameters():
param.requires_grad = False




def print_num_params(model, only_trainable=False):
total = 0
trainable = 0
for p in model.parameters():
num = p.numel()
total += num
if p.requires_grad:
trainable += num
if only_trainable:
print(f"Trainable params: {trainable}")
else:
print(f"Total params: {total}, Trainable: {trainable}")




def safe_device_map():
# Placeholder: let accelerate or device_map='auto' handle this in scripts
return None
