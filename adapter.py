import torch
from torch import nn

class Adapter(nn.Module):
    def __init__(self, input_size, bottleneck_size):
        super().__init__()
        self.down = nn.Linear(input_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, input_size)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.up(self.act(self.down(x))) + x

class AdapterManager:
    def __init__(self):
        self.adapters = {}

    def create_adapter(self, layer_name, input_size, bottleneck_size=64):
        self.adapters[layer_name] = Adapter(input_size, bottleneck_size)

    def apply_adapter(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.adapters:
                adapter = self.adapters[name]
                original_forward = module.forward
                
                def new_forward(x):
                    return adapter(original_forward(x))
                
                module.forward = new_forward
        
        return model

    def train_adapters(self, model, data_loader, optimizer, num_epochs):
        for epoch in range(num_epochs):
            for batch in data_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
