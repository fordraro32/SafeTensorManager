from flask import Flask, render_template, request, jsonify
from model_handler import ModelHandler
from adapter import AdapterManager
from memory_manager import MemoryManager
from gpu_utils import setup_distributed, cleanup

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
    
    return jsonify({'average_loss': sum(losses) / len(losses)})

if __name__ == '__main__':
    setup_distributed()
    app.run(host='0.0.0.0', port=5000, debug=True)
    cleanup()
