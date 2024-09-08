import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelHandler:
    def __init__(self):
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

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
