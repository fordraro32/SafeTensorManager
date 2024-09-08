from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_code():
    prompt = request.json['prompt']
    # Simulate code generation
    generated_code = f"# Generated code for: {prompt}\n\ndef example_function():\n    print('Hello, World!')\n\nexample_function()"
    return jsonify({'generated_code': generated_code})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
