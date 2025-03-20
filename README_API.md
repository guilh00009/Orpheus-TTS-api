# Orpheus TTS API

A Python Flask API for the Orpheus Text-to-Speech system.

## Features

- Text-to-speech conversion with multiple voice options
- Support for emotional tags (laugh, sigh, etc.)
- Streaming audio output
- Downloadable WAV files
- Web interface for easy testing

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/orpheus-tts-api.git
   cd orpheus-tts-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with vllm, you can downgrade to a stable version:
   ```bash
   pip install vllm==0.7.3
   ```

3. Run the API:
   ```bash
   python app.py
   ```

   For production deployment:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8080 app:app
   ```

4. Access the web interface at: http://localhost:8080

## API Endpoints

### Web Interface
- `GET /` - Web interface for testing the TTS service
- `GET /docs` - API documentation

### TTS Endpoints
- `GET /api/tts/stream` - Stream audio in real-time
  - Query parameters:
    - `text` (required): Text to convert to speech
    - `voice` (optional): Voice to use (default: "tara")
    - `temperature` (optional): Generation temperature (default: 0.4)
    - `repetition_penalty` (optional): Repetition penalty (default: 1.1)

- `POST /api/tts/generate` - Generate and download WAV file
  - JSON body parameters:
    - `text` (required): Text to convert to speech
    - `voice` (optional): Voice to use (default: "tara")
    - `emotions` (optional): Array of emotion tags to apply
    - `temperature` (optional): Generation temperature (default: 0.4)
    - `repetition_penalty` (optional): Repetition penalty (default: 1.1)
    - `max_tokens` (optional): Maximum tokens (default: 2000)
    - `top_p` (optional): Top p value (default: 0.9)

### Information Endpoints
- `GET /api/voices` - Get list of available voices
- `GET /api/emotions` - Get list of available emotion tags

## Available Voices

- tara
- leah
- jess
- leo
- dan
- mia
- zac
- zoe

## Available Emotion Tags

- laugh
- chuckle
- sigh
- cough
- sniffle
- groan
- yawn
- gasp

## Examples

### Streaming TTS with curl
```bash
curl "http://localhost:8080/api/tts/stream?text=Hello%20world&voice=tara" -o output.wav
```

### Generating a WAV file with Python
```python
import requests
import json

url = "http://localhost:8080/api/tts/generate"
data = {
    "text": "Hello, this is a test of the Orpheus TTS system.",
    "voice": "leah",
    "emotions": ["laugh"],
    "temperature": 0.5,
    "repetition_penalty": 1.2
}

response = requests.post(url, json=data)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Client Integration

For client-side integration, here's a simple JavaScript example:

```javascript
// Stream audio
const text = "Hello world";
const voice = "tara";
const streamUrl = `/api/tts/stream?text=${encodeURIComponent(text)}&voice=${voice}`;

const audio = new Audio(streamUrl);
audio.play();

// Generate and download
async function generateAndDownload() {
    const response = await fetch('/api/tts/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: "Hello world",
            voice: "tara",
            emotions: ["laugh"]
        })
    });
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'tts_audio.wav';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
}
```

## License

This project is licensed under the terms specified in the Orpheus TTS project. 