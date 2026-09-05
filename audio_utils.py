"""
Small helpers for turning a numpy audio array into downloadable bytes,
either as .wav (lossless) or .mp3 (smaller, easier to share), and for
decoding uploaded audio (wav/mp3/m4a/flac/ogg) into a clean format the
model can use.
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


def _read_with_soundfile(raw_bytes: bytes):
    """Try reading audio bytes directly with soundfile (fast path)."""
    import soundfile as sf

    return sf.read(io.BytesIO(raw_bytes))


def _read_with_ffmpeg(raw_bytes: bytes):
    """
    Fallback for formats soundfile can't read directly (e.g. m4a/AAC
    from iPhone voice memos). Uses pydub, which shells out to ffmpeg,
    to transcode to WAV first, then reads that with soundfile.
    """
    import soundfile as sf
    from pydub import AudioSegment

    audio_segment = AudioSegment.from_file(io.BytesIO(raw_bytes))
    wav_buf = io.BytesIO()
    audio_segment.export(wav_buf, format="wav")
    wav_buf.seek(0)

    return sf.read(wav_buf)


def read_any_audio(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Take uploaded audio bytes in any common format (wav, mp3, m4a,
    flac, ogg) and return (audio_samples, sample_rate) as soundfile
    would. Tries soundfile first; falls back to ffmpeg/pydub for
    formats soundfile can't handle natively (notably m4a/AAC).
    """
    try:
        return _read_with_soundfile(raw_bytes)
    except Exception:
        return _read_with_ffmpeg(raw_bytes)


def decode_any_to_wav_bytes(raw_bytes: bytes) -> bytes:
    """
    Take uploaded audio bytes in *any* format we support and return
    proper WAV bytes.
    """
    data, sample_rate = read_any_audio(raw_bytes)
    return to_wav_bytes(data, sample_rate)
