from flask import Flask, Response, request, jsonify, send_file, render_template_string, render_template
import os
import struct
import uuid
import wave
import time
import io
import torch
import numpy as np
import traceback

app = Flask(__name__, template_folder='templates')

# Flag to track if models have been loaded
vozia_model_loaded = False
orpheus_model_loaded = False

# Model variables
tokenizer = None
ori_model = None
snac_model = None
orpheus_engine = None

# Available voices for Orpheus model
ORPHEUS_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

# Available emotion tags that can be embedded in text
EMOTION_TAGS = ["<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>", "<groan>", "<yawn>", "<gasp>"]

def load_vozia_model():
    """Load the Vozia model (Guilherme34/Vozia-3b-lora) and its dependencies."""
    global vozia_model_loaded, tokenizer, ori_model, snac_model
    
    if vozia_model_loaded:
        return True
    
    try:
        # Set CUDA device if needed
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Import necessary modules for Vozia model
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from snac import SNAC
        
        print("Downloading Vozia-3b-lora model...")
        # Download the model files
        model_path = snapshot_download(repo_id="Guilherme34/Vozia-3b-lora", 
                        allow_patterns=["*.safetensors", "*.json", "*.md"],
                        local_dir='./vozia_model')
        
        print(f"Model downloaded to {model_path}")
        print("Files in model directory:", os.listdir(model_path))
        
        print("Loading base model and tokenizer...")
        # Load the base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('canopylabs/orpheus-3b-0.1-ft')
        ori_model = AutoModelForCausalLM.from_pretrained(
            'canopylabs/orpheus-3b-0.1-ft', 
            tie_word_embeddings=False
        )
        
        if torch.cuda.is_available():
            ori_model = ori_model.cuda()
        
        # Apply embedding weights to lm_head
        ori_model.lm_head.weight.data = ori_model.model.embed_tokens.weight.data.clone()
        
        print("Loading and applying LoRA adapter weights...")
        # Get all safetensors files in the model directory
        model_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        print(f"Found model files: {model_files}")
        
        if not model_files:
            raise FileNotFoundError("No .safetensors files found in downloaded model directory")
            
        # Process each model file - apply LoRA weights as deltas
        for model_file in model_files:
            file_path = os.path.join(model_path, model_file)
            print(f"Loading weights from: {file_path}")
            
            # Properly apply LoRA weights as deltas
            adapter_weights = load_file(file_path)
            
            # Get the model state dict that will be modified
            state_dict = ori_model.state_dict()
            
            # Find all LoRA keys in the adapter weights
            lora_keys = [k for k in adapter_weights.keys() if '.lora_' in k]
            base_keys = sorted(list(set([k.split('.lora')[0] for k in lora_keys])))
            
            print(f"Applying {len(base_keys)} LoRA modules")
            
            # Apply LoRA weights to the base model
            for key in base_keys:
                if 'embed_tokens' in key:
                    lora_a = key + '.lora_embedding_A'
                    lora_b = key + '.lora_embedding_B'
                else:
                    lora_a = key + '.lora_A.weight'
                    lora_b = key + '.lora_B.weight'
                
                # Get the target key (removing 'base_model.model.' prefix if present)
                target_key = key.replace('base_model.model.', '') + '.weight'
                if target_key not in state_dict:
                    print(f"Warning: Target key {target_key} not found in model")
                    continue
                
                # Get the weights
                if lora_a in adapter_weights and lora_b in adapter_weights:
                    a_weight = adapter_weights[lora_a].to(ori_model.device)
                    b_weight = adapter_weights[lora_b].to(ori_model.device)
                    
                    # Get the original weight and prepare it for update
                    weight = state_dict[target_key]
                    
                    # For non-embedding layers, we need to transpose
                    if 'embed_tokens' not in key:
                        weight = weight.t()
                    
                    # Apply LoRA formula: W = W + BA (with scaling)
                    with torch.no_grad():
                        scaling = 1.5  # Common LoRA scaling factor
                        delta = torch.matmul(b_weight.t(), a_weight.t()) * scaling
                        
                        if delta.shape == weight.shape:
                            weight.add_(delta)
                        else:
                            print(f"Warning: Shape mismatch for {target_key}")
                            print(f"Delta shape: {delta.shape}, Weight shape: {weight.shape}")
                            
                    # Transpose back for non-embedding layers
                    if 'embed_tokens' not in key:
                        weight = weight.t()
        
        print("Loading SNAC model...")
        # Load SNAC model for audio decoding
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        if torch.cuda.is_available():
            snac_model = snac_model.to("cuda")
        
        vozia_model_loaded = True
        print("Vozia model loaded successfully!")
        return True
    
    except Exception as e:
        print(f"Error loading Vozia model: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        return False

def load_orpheus_model():
    """Load the standard Orpheus TTS model."""
    global orpheus_model_loaded, orpheus_engine
    
    if orpheus_model_loaded:
        return True
    
    try:
        from orpheus_tts import OrpheusModel
        orpheus_engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        orpheus_model_loaded = True
        return True
    except Exception as e:
        print(f"Error loading Orpheus model: {str(e)}")
        return False

def redistribute_codes(row):
    """Redistribute codes for audio generation with SNAC model."""
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = row[:new_length]
    code_list = [t - 128266 for t in trimmed_row]
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i][None])
        layer_2.append(code_list[7*i+1][None]-4096)
        layer_3.append(code_list[7*i+2][None]-(2*4096))
        layer_3.append(code_list[7*i+3][None]-(3*4096))
        layer_2.append(code_list[7*i+4][None]-(4*4096))
        layer_3.append(code_list[7*i+5][None]-(5*4096))
        layer_3.append(code_list[7*i+6][None]-(6*4096))
    
    with torch.no_grad():
        codes = [torch.concat(layer_1)[None], 
                torch.concat(layer_2)[None], 
                torch.concat(layer_3)[None]]
        audio_hat = snac_model.decode(codes)
        return audio_hat.cpu()[0, 0]

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       
        b'WAVE',
        b'fmt ',
        16,                  
        1,             
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TTS API Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            pre { background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .endpoint { margin-bottom: 30px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .note { background-color: #e9f7fe; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #4CAF50; }
            .model-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; background-color: #4CAF50; color: white; font-size: 12px; margin-left: 10px; }
        </style>
    </head>
    <body>
        <h1>TTS API Documentation</h1>
        
        <div class="note">
            <strong>Main Model:</strong> This API primarily uses the <strong>Guilherme34/Vozia-3b-lora</strong> model.
            <br><br>
            <strong>Emotion Tags:</strong> You can add emotion tags directly in your text to control the speech. 
            For example: "Hi <laugh> how are you today?" or "I can't believe it <sigh>".
            <br><br>
            Available tags: &lt;laugh&gt;, &lt;chuckle&gt;, &lt;sigh&gt;, &lt;cough&gt;, &lt;sniffle&gt;, &lt;groan&gt;, &lt;yawn&gt;, &lt;gasp&gt;
        </div>
        
        <div class="endpoint">
            <h2>Generate TTS <span class="model-badge">Vozia Model</span></h2>
            <p>Generate and download a WAV file using the Vozia model</p>
            <pre>POST /api/tts/generate</pre>
            
            <h3>JSON Body Parameters:</h3>
            <table>
                <tr><th>Name</th><th>Type</th><th>Description</th></tr>
                <tr><td>text</td><td>string</td><td>Required. The text to convert to speech. Can include emotion tags like &lt;laugh&gt;.</td></tr>
                <tr><td>speaker</td><td>string</td><td>Optional. Speaker name to use (default: "Speaker")</td></tr>
                <tr><td>temperature</td><td>float</td><td>Optional. Generation temperature (default: 0.9)</td></tr>
                <tr><td>repetition_penalty</td><td>float</td><td>Optional. Repetition penalty (default: 1.1)</td></tr>
                <tr><td>max_tokens</td><td>integer</td><td>Optional. Maximum number of tokens (default: 1200)</td></tr>
                <tr><td>top_p</td><td>float</td><td>Optional. Top p value (default: 0.95)</td></tr>
            </table>
        </div>
        
        <div class="endpoint">
            <h2>Stream TTS <span class="model-badge">Vozia Model</span></h2>
            <p>Stream audio from text in real-time</p>
            <pre>GET /api/tts/stream?text=Hello&speaker=Speaker</pre>
            
            <h3>Parameters:</h3>
            <table>
                <tr><th>Name</th><th>Type</th><th>Description</th></tr>
                <tr><td>text</td><td>string</td><td>Required. The text to convert to speech. Can include emotion tags like &lt;laugh&gt;.</td></tr>
                <tr><td>speaker</td><td>string</td><td>Optional. Speaker name to use (default: "Speaker")</td></tr>
                <tr><td>temperature</td><td>float</td><td>Optional. Generation temperature (default: 0.9)</td></tr>
                <tr><td>repetition_penalty</td><td>float</td><td>Optional. Repetition penalty (default: 1.1)</td></tr>
            </table>
        </div>
        
        <div class="endpoint">
            <h2>Generate with Orpheus <span class="model-badge">Orpheus Model</span></h2>
            <p>Generate and download a WAV file using the original Orpheus model</p>
            <pre>POST /api/orpheus/generate</pre>
            
            <h3>JSON Body Parameters:</h3>
            <table>
                <tr><th>Name</th><th>Type</th><th>Description</th></tr>
                <tr><td>text</td><td>string</td><td>Required. The text to convert to speech. Can include emotion tags like &lt;laugh&gt;.</td></tr>
                <tr><td>voice</td><td>string</td><td>Optional. Voice to use (default: "tara")</td></tr>
                <tr><td>temperature</td><td>float</td><td>Optional. Generation temperature (default: 0.4)</td></tr>
                <tr><td>repetition_penalty</td><td>float</td><td>Optional. Repetition penalty (default: 1.1)</td></tr>
                <tr><td>max_tokens</td><td>integer</td><td>Optional. Maximum number of tokens (default: 2000)</td></tr>
                <tr><td>top_p</td><td>float</td><td>Optional. Top p value (default: 0.9)</td></tr>
            </table>
        </div>
        
        <div class="endpoint">
            <h2>Stream with Orpheus <span class="model-badge">Orpheus Model</span></h2>
            <p>Stream audio using the original Orpheus model</p>
            <pre>GET /api/orpheus/stream?text=Hello&voice=tara</pre>
            
            <h3>Parameters:</h3>
            <table>
                <tr><th>Name</th><th>Type</th><th>Description</th></tr>
                <tr><td>text</td><td>string</td><td>Required. The text to convert to speech. Can include emotion tags like &lt;laugh&gt;.</td></tr>
                <tr><td>voice</td><td>string</td><td>Optional. Voice to use (default: "tara")</td></tr>
                <tr><td>temperature</td><td>float</td><td>Optional. Generation temperature (default: 0.4)</td></tr>
                <tr><td>repetition_penalty</td><td>float</td><td>Optional. Repetition penalty (default: 1.1)</td></tr>
            </table>
        </div>
        
        <div class="endpoint">
            <h2>Orpheus Voices</h2>
            <p>Get list of available voices for the Orpheus model</p>
            <pre>GET /api/orpheus/voices</pre>
        </div>
        
        <div class="endpoint">
            <h2>Emotion Tags</h2>
            <p>Get list of available emotion tags that can be embedded in the text</p>
            <pre>GET /api/emotion-tags</pre>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api/tts/stream')
def stream_tts():
    """Stream TTS using the Vozia model."""
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "Text parameter is required"}), 400
    
    # Load the Vozia model if not already loaded
    if not load_vozia_model():
        return jsonify({"error": "Failed to load Vozia model. Check server logs for details."}), 500
    
    speaker = request.args.get('speaker', 'Speaker')
    temperature = float(request.args.get('temperature', '0.9'))
    repetition_penalty = float(request.args.get('repetition_penalty', '1.1'))
    
    # Create the prompt
    prompt = f'<custom_token_3><|begin_of_text|>{speaker}: {text}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>'
    
    try:
        # Process input
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
        
        # Generate
        with torch.no_grad():
            generated_ids = ori_model.generate(
                **input_ids,
                max_new_tokens=1200,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=128258,
            )
        
        # Process the generated audio
        row = generated_ids[0, input_ids['input_ids'].shape[1]:]
        audio_data = redistribute_codes(row)
        
        # Convert to bytes
        audio_data = audio_data.numpy()
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Define chunk size (e.g., 0.5 seconds of audio)
        sample_rate = 24000
        chunk_size = int(sample_rate * 0.5) * 2  # 0.5 seconds of 16-bit audio
        
        # Create a generator to yield chunks of audio
        def generate_audio_stream():
            # First yield the WAV header
            yield create_wav_header()
            
            # Yield the audio data in chunks
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                if chunk:
                    yield chunk
                    # Add a small delay to simulate realistic streaming
                    time.sleep(0.1)

        return Response(generate_audio_stream(), mimetype='audio/wav')
    
    except Exception as e:
        return jsonify({"error": f"Error generating speech: {str(e)}"}), 500

@app.route('/api/tts/generate', methods=['POST'])
def generate_tts():
    """Generate TTS using the Vozia model."""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Text parameter is required"}), 400
    
    # Load the Vozia model if not already loaded
    if not load_vozia_model():
        return jsonify({"error": "Failed to load Vozia model. Check server logs for details."}), 500
    
    text = data['text']
    speaker = data.get('speaker', 'Speaker')
    
    # Get generation parameters
    temperature = float(data.get('temperature', 0.9))
    repetition_penalty = float(data.get('repetition_penalty', 1.1))
    max_tokens = int(data.get('max_tokens', 1200))
    top_p = float(data.get('top_p', 0.95))
    
    # Create the prompt
    prompt = f'<custom_token_3><|begin_of_text|>{speaker}: {text}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>'
    
    try:
        # Create input_ids
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
        
        # Generate
        with torch.no_grad():
            generated_ids = ori_model.generate(
                **input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=128258,
            )
        
        # Process the generated audio
        row = generated_ids[0, input_ids['input_ids'].shape[1]:]
        audio_data = redistribute_codes(row)
        
        # Convert to bytes
        audio_data = audio_data.numpy()
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Create WAV file
        output_buffer = io.BytesIO()
        with wave.open(output_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_bytes)
        
        # Reset buffer position
        output_buffer.seek(0)
        
        # Generate a unique filename
        filename = f"vozia_tts_{uuid.uuid4().hex[:8]}.wav"
        
        # Return the audio file
        return send_file(
            output_buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({"error": f"Error generating speech: {str(e)}"}), 500

@app.route('/api/orpheus/stream')
def stream_orpheus_tts():
    """Stream audio using the original Orpheus model."""
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "Text parameter is required"}), 400
    
    # Load the Orpheus model if not already loaded
    if not load_orpheus_model():
        return jsonify({"error": "Failed to load Orpheus model. Check server logs for details."}), 500
    
    voice = request.args.get('voice', 'tara')
    if voice not in ORPHEUS_VOICES:
        return jsonify({"error": f"Invalid voice. Available voices: {', '.join(ORPHEUS_VOICES)}"}), 400
    
    temperature = float(request.args.get('temperature', '0.4'))
    repetition_penalty = float(request.args.get('repetition_penalty', '1.1'))
    
    def generate_audio_stream():
        yield create_wav_header()

        syn_tokens = orpheus_engine.generate_speech(
            prompt=text,
            voice=voice,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[128258],
            max_tokens=2000,
            temperature=temperature,
            top_p=0.9
        )
        for chunk in syn_tokens:
            yield chunk

    return Response(generate_audio_stream(), mimetype='audio/wav')

@app.route('/api/orpheus/generate', methods=['POST'])
def generate_orpheus_tts():
    """Generate TTS using the original Orpheus model."""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Text parameter is required"}), 400
    
    # Load the Orpheus model if not already loaded
    if not load_orpheus_model():
        return jsonify({"error": "Failed to load Orpheus model. Check server logs for details."}), 500
    
    text = data['text']
    voice = data.get('voice', 'tara')
    
    if voice not in ORPHEUS_VOICES:
        return jsonify({"error": f"Invalid voice. Available voices: {', '.join(ORPHEUS_VOICES)}"}), 400
    
    # Get generation parameters
    temperature = float(data.get('temperature', 0.4))
    repetition_penalty = float(data.get('repetition_penalty', 1.1))
    max_tokens = int(data.get('max_tokens', 2000))
    top_p = float(data.get('top_p', 0.9))
    
    # Create an in-memory file-like object
    output_buffer = io.BytesIO()
    
    with wave.open(output_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        
        start_time = time.monotonic()
        syn_tokens = orpheus_engine.generate_speech(
            prompt=text,
            voice=voice,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[128258],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        total_frames = 0
        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
        
        end_time = time.monotonic()
        duration = total_frames / wf.getframerate()
    
    # Reset the buffer position to the start
    output_buffer.seek(0)
    
    # Generate a unique filename
    filename = f"orpheus_tts_{uuid.uuid4().hex[:8]}.wav"
    
    # Return the audio file as an attachment
    return send_file(
        output_buffer,
        mimetype="audio/wav",
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/orpheus/voices')
def get_orpheus_voices():
    """Get list of available voices for the Orpheus model."""
    return jsonify({
        "voices": [
            {"id": voice, "name": voice.capitalize()} for voice in ORPHEUS_VOICES
        ]
    })

@app.route('/api/emotion-tags')
def get_emotion_tags():
    """Get list of available emotion tags."""
    return jsonify({
        "emotionTags": EMOTION_TAGS
    })

if __name__ == '__main__':
    # Pre-load the Vozia model on startup
    print("Pre-loading Vozia model...")
    load_vozia_model()
    
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, threaded=True) 
