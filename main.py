from flask import Flask, render_template, request, jsonify
from model_handler import ModelHandler
from adapter import AdapterManager
from memory_manager import MemoryManager
from gpu_utils import setup_distributed, cleanup
import torch

app = Flask(__name__)

model_handler = ModelHandler()
adapter_manager = AdapterManager()
memory_manager = MemoryManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_code():
    prompt = request.json['prompt']
    context = memory_manager.get_context(prompt)
    generated_code = model_handler.generate(prompt, context, adapter_manager)
    memory_manager.update_context(prompt, generated_code)
    return jsonify({'generated_code': generated_code})

@app.route('/train', methods=['POST'])
def train_model():
    num_steps = request.json.get('num_steps', 100)
    optimizer = torch.optim.Adam(model_handler.model.parameters(), lr=1e-4)
    
    losses = []
    for _ in range(num_steps):
        loss = model_handler.train_step(optimizer)
        losses.append(loss)
    
    # Optimize memory after training
    model_handler.optimize_memory()
    
    return jsonify({'average_loss': sum(losses) / len(losses)})

@app.route('/cuda_graph_demo', methods=['POST'])
def cuda_graph_demo():
    input_size = request.json.get('input_size', 1024)
    input_data = torch.randn(input_size, device='cuda')
    
    # Run inference using CUDA graph
    output = model_handler.run_with_cuda_graph(input_data)
    
    return jsonify({'output_shape': list(output.shape), 'output_mean': output.mean().item()})

@app.route('/memory_stats', methods=['GET'])
def memory_stats():
    return jsonify({
        'total_memory': model_handler.memory_manager.max_memory,
        'used_memory': model_handler.memory_manager.current_memory,
        'available_memory': model_handler.memory_manager.max_memory - model_handler.memory_manager.current_memory
    })

@app.route('/distributed_train', methods=['POST'])
def distributed_train():
    num_steps = request.json.get('num_steps', 100)
    optimizer = torch.optim.Adam(model_handler.model.parameters(), lr=1e-4)
    
    losses = []
    for _ in range(num_steps):
        loss = model_handler.train_step(optimizer)
        losses.append(loss)
    
    # Simulate distributed synchronization
    model_handler.distributed_context.barrier()
    
    return jsonify({
        'average_loss': sum(losses) / len(losses),
        'world_size': model_handler.distributed_context.world_size,
        'rank': model_handler.distributed_context.rank
    })

if __name__ == '__main__':
    setup_distributed()
    app.run(host='0.0.0.0', port=5000, debug=True)
    cleanup()
