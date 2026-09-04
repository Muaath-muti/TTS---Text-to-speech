# Voice of Good Hope — University TTS Tool

A text-to-speech app built on [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts).
Runs entirely on CPU, no cloud API keys needed.

🔗 **[Try the live app here](https://voice-of-good-hope.streamlit.app)**

## What's new in this version

- **MP3 in, MP3 out.** Upload mp3/m4a/flac/ogg/wav for voice cloning, and
  download generated speech as either .wav or .mp3.
- **Redesigned interface** — custom styling, clearer layout, three tabs
  (Generate speech / Voice library / Add a new voice).
- **Voice library tab** — see every built-in and cloned voice in one
  place, delete cloned voices you don't need anymore.
- **Generation history** — your last 10 generations stay available to
  replay or re-download without regenerating.
- **Live word count + estimated duration** while you type.
- **Stability controls** in the sidebar — if a voice breaks up or
  repeats on longer text, raise the "end-of-sentence sensitivity"
  slider and try again.

## Files

- `app.py` — the Streamlit interface
- `tts_engine.py` — wraps the `pocket-tts` library: loads the model,
  generates speech, clones/saves/deletes voices
- `audio_utils.py` — converts audio to downloadable .wav / .mp3 bytes,
  and decodes any uploaded format into clean .wav for the model
- `voices/` — cloned voices are saved here as `.safetensors` files

## Setup

1. Install Python 3.10–3.14.
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
   Activate it:
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
     (if blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)
   - Mac/Linux: `source .venv/bin/activate`
3. Install dependencies:
   ```
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   (On macOS/Windows you can drop the `--extra-index-url` part.)

## Running it

```
streamlit run app.py
```

Opens a browser tab at `http://localhost:8501`. First generation
downloads the model weights (~200MB) from Hugging Face — needs
internet the first time only.

### Voice cloning requires a one-time Hugging Face login

Pocket TTS's voice-cloning model requires accepting terms on Hugging
Face before first use:

1. Create a free account at [huggingface.co/join](https://huggingface.co/join)
2. Accept the terms at [huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
3. Get a token (Read access) at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. In your terminal (with the venv active), run:
   ```
   hf auth login
   ```
   and paste the token when prompted.

## Adding your South African voice

1. Record (or find) a clean 10–30 second sample — one speaker, quiet
   room, no background noise. Any format works (wav, mp3, m4a, flac, ogg).
2. Go to the **"Add a new voice"** tab, name it, upload the sample,
   click **Clone and save this voice**.
3. It appears in the voice dropdown and the **Voice library** tab from
   then on.

If a cloned voice breaks up or repeats on longer text, it's usually
because the source audio is lossy/noisy — re-clone from the cleanest
source you can, and/or raise the "end-of-sentence sensitivity" slider
in the sidebar.

## Moving to a server later

Nothing here is laptop-specific. On a university server:
```
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
(behind your usual auth/reverse proxy), and multiple people can use it
from their browsers at once.
