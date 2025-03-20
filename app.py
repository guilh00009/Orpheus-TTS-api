from flask import Flask, Response, request, jsonify, send_file, render_template_string, render_template
import os
import struct
import uuid
import wave
import time
import io
from orpheus_tts import OrpheusModel

app = Flask(__name__, template_folder='templates')

# Initialize the Orpheus TTS model
engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

# Available voices
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

# Available emotion tags that can be embedded in text
EMOTION_TAGS = ["<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>", "<groan>", "<yawn>", "<gasp>"]

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
        <title>Orpheus TTS API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            pre { background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .endpoint { margin-bottom: 30px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .note { background-color: #e9f7fe; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #4CAF50; }
        </style>
    </head>
    <body>
        <h1>Orpheus TTS API Documentation</h1>
        
        <div class="note">
            <strong>Emotion Tags:</strong> You can add emotion tags directly in your text to control the speech. 
            For example: "Hi <laugh> how are you today?" or "I can't believe it <sigh>".
            <br><br>
            Available tags: &lt;laugh&gt;, &lt;chuckle&gt;, &lt;sigh&gt;, &lt;cough&gt;, &lt;sniffle&gt;, &lt;groan&gt;, &lt;yawn&gt;, &lt;gasp&gt;
        </div>
        
        <div class="endpoint">
            <h2>Stream TTS</h2>
            <p>Stream audio from text in real-time</p>
            <pre>GET /api/tts/stream?text=Hello&voice=tara</pre>
            
            <h3>Parameters:</h3>
            <table>
                <tr><th>Name</th><th>Type</th><th>Description</th></tr>
                <tr><td>text</td><td>string</td><td>Required. The text to convert to speech. Can include emotion tags like &lt;laugh&gt;.</td></tr>
                <tr><td>voice</td><td>string</td><td>Optional. Voice to use (default: tara)</td></tr>
                <tr><td>temperature</td><td>float</td><td>Optional. Generation temperature (default: 0.4)</td></tr>
                <tr><td>repetition_penalty</td><td>float</td><td>Optional. Repetition penalty (default: 1.1)</td></tr>
            </table>
        </div>
        
        <div class="endpoint">
            <h2>Generate Audio File</h2>
            <p>Generate and download a WAV file</p>
            <pre>POST /api/tts/generate</pre>
            
            <h3>JSON Body Parameters:</h3>
            <table>
                <tr><th>Name</th><th>Type</th><th>Description</th></tr>
                <tr><td>text</td><td>string</td><td>Required. The text to convert to speech. Can include emotion tags like &lt;laugh&gt;.</td></tr>
                <tr><td>voice</td><td>string</td><td>Optional. Voice to use (default: tara)</td></tr>
                <tr><td>temperature</td><td>float</td><td>Optional. Generation temperature (default: 0.4)</td></tr>
                <tr><td>repetition_penalty</td><td>float</td><td>Optional. Repetition penalty (default: 1.1)</td></tr>
                <tr><td>max_tokens</td><td>integer</td><td>Optional. Maximum number of tokens (default: 2000)</td></tr>
                <tr><td>top_p</td><td>float</td><td>Optional. Top p value (default: 0.9)</td></tr>
            </table>
        </div>
        
        <div class="endpoint">
            <h2>Voices</h2>
            <p>Get list of available voices</p>
            <pre>GET /api/voices</pre>
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
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "Text parameter is required"}), 400
    
    voice = request.args.get('voice', 'tara')
    if voice not in VOICES:
        return jsonify({"error": f"Invalid voice. Available voices: {', '.join(VOICES)}"}), 400
    
    temperature = float(request.args.get('temperature', '0.4'))
    repetition_penalty = float(request.args.get('repetition_penalty', '1.1'))
    
    def generate_audio_stream():
        yield create_wav_header()

        syn_tokens = engine.generate_speech(
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

@app.route('/api/tts/generate', methods=['POST'])
def generate_tts():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Text parameter is required"}), 400
    
    text = data['text']
    voice = data.get('voice', 'tara')
    
    if voice not in VOICES:
        return jsonify({"error": f"Invalid voice. Available voices: {', '.join(VOICES)}"}), 400
    
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
        syn_tokens = engine.generate_speech(
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

@app.route('/api/voices')
def get_voices():
    return jsonify({
        "voices": [
            {"id": voice, "name": voice.capitalize()} for voice in VOICES
        ]
    })

@app.route('/api/emotion-tags')
def get_emotion_tags():
    return jsonify({
        "emotionTags": EMOTION_TAGS
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, threaded=True) 