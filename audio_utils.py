"""
Small helpers for turning a numpy audio array into downloadable bytes,
either as .wav (lossless) or .mp3 (smaller, easier to share).
"""

from __future__ import annotations

import io

import lameenc
import numpy as np
import scipy.io.wavfile as wavfile


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float audio array as standard 16-bit PCM WAV bytes."""
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, pcm16)
    return buf.getvalue()


def to_mp3_bytes(audio: np.ndarray, sample_rate: int, bitrate: int = 192) -> bytes:
    """Encode a float audio array as MP3 bytes (no ffmpeg required)."""
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 2 = high quality, still fast enough for our sizes
    mp3_data = encoder.encode(pcm16.tobytes())
    mp3_data += encoder.flush()
    # lameenc returns a bytearray, but Streamlit's download_button (and
    # some other consumers) require real bytes — convert explicitly.
    return bytes(mp3_data)


def decode_any_to_wav_bytes(raw_bytes: bytes) -> bytes:
    """
    Take uploaded audio bytes in *any* format soundfile understands
    (wav, mp3, m4a, flac, ogg) and return proper WAV bytes.
    """
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(raw_bytes))
    return to_wav_bytes(data, sample_rate)
