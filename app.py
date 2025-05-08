from flask import Flask, render_template, request, jsonify
import subprocess
import os
import json
import re
import requests

app = Flask(__name__)

os.makedirs(os.path.join('static', 'css'), exist_ok=True)
os.makedirs(os.path.join('static', 'js'), exist_ok=True)

@app.route('/')
def index():
    config = {
        'backend_url': 'http://localhost:8080',
        'duration': '10'
    }
    return render_template('identify.html', config=config)

@app.route('/api/test', methods=['GET'])
def test_connection():
    return jsonify({
        'status': 'success',
        'message': 'Connection successful from backend server!'
    })

@app.route('/api/identify', methods=['POST'])
def identify_song():
    data = request.json
    audio_file_path = data.get('audioFilePath', '')
    duration = data.get('duration', '10')
    
    if not audio_file_path:
        return jsonify({
            'status': 'error',
            'message': 'Please enter the path to an audio file.'
        }), 400
    
    if not os.path.exists(audio_file_path):
        return jsonify({
            'status': 'error',
            'message': f'Audio file not found: {audio_file_path}'
        }), 404
    
    cmd = [
        'python3',
        'frontend_fingerprinting.py',
        '--identify', audio_file_path,
        '--duration', duration
    ]
    
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = process.stdout + process.stderr
        
        identification_results = None
        try:
            # Attempt to extract a JSON object from the script output using a regular expression.
            #   This is necessary because the script may print other text.
            match = re.search(r'({.*})', output, re.DOTALL)
            if match:
                json_text = match.group(1)
                identification_results = json.loads(json_text)
        except Exception as e:
            print(f"Error parsing identification results: {e}")
        
        return jsonify({
            'status': 'success',
            'console_output': output,
            'identification_results': identification_results
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({
            'status': 'error',
            'message': 'Script execution timed out',
            'console_output': 'The script took too long to execute and was terminated.'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'console_output': str(e)
        }), 500

@app.route('/api/save_config', methods=['POST'])
def save_config():
    return jsonify({
        'status': 'success',
        'message': 'Configuration saved'
    })

@app.route('/api/search', methods=['GET'])
def search_songs():
    query = request.args.get('query', '')
    
    if not query:
        return jsonify({
            'status': 'error',
            'message': 'Missing search query'
        }), 400
    
    backend_url = request.args.get('backend_url', 'http://localhost:8080')
    
    try:

        response = requests.get(f"{backend_url}/api/search", params={'query': query})
        return jsonify(response.json()), response.status_code
        
    except requests.RequestException as e:
        return jsonify({
            'status': 'error',
            'message': f'Error connecting to backend: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='localhost')