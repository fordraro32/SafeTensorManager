import torch

class MemoryManager:
    def __init__(self, max_context_length=10000):
        self.context = ""
        self.max_context_length = max_context_length

    def get_context(self, prompt):
        return self.context[-self.max_context_length:] + prompt

    def update_context(self, prompt, generated_code):
        self.context += f"\nPrompt: {prompt}\nGenerated Code:\n{generated_code}\n"
        self.context = self.context[-self.max_context_length:]

    @staticmethod
    def offload_to_cpu(tensor):
        return tensor.to('cpu')

    @staticmethod
    def load_to_gpu(tensor):
        return tensor.to('cuda')
