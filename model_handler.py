import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpu_utils import magnum_io_data_loader

class ModelHandler:
    def __init__(self):
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        self.data_loader = magnum_io_data_loader(batch_size=32, num_threads=4, device_id=0)

    def load_model(self):
        model_path = "codellama/CodeLlama-7b-hf"
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
        return model

    def generate(self, prompt, context, adapter_manager):
        # Combine prompt with context
        full_prompt = f"{context}\n{prompt}"
        
        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        # Apply adapter
        adapted_model = adapter_manager.apply_adapter(self.model)
        
        # Generate
        with torch.no_grad():
            outputs = adapted_model.generate(**inputs, max_length=1000, num_return_sequences=1)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def train_step(self, optimizer):
        images, labels = next(self.data_loader)
        
        # Move data to GPU
        images = images.to(self.model.device)
        labels = labels.to(self.model.device)

        # Forward pass
        outputs = self.model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
