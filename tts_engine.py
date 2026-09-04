"""
Thin wrapper around the `pocket-tts` library (Kyutai).

Handles:
- loading the model once and reusing it
- generating speech from text + a chosen voice
- cloning a new voice from an uploaded audio sample (any format
  soundfile understands) and saving it to disk for reuse
- listing / deleting voices in the local voice library
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pocket_tts import TTSModel, export_model_state

VOICES_DIR = Path(__file__).parent / "voices"
VOICES_DIR.mkdir(exist_ok=True)

# Pre-made voices that ship with Pocket TTS itself.
BUILT_IN_VOICES: dict[str, str] = {
    "Alba (English)": "alba",
    "Cosette (English)": "cosette",
    "Marius (English)": "marius",
    "Javert (English)": "javert",
    "Jean (English)": "jean",
}


class TTSEngine:
    """Loads the Pocket TTS model once and exposes simple generate/clone calls."""

    def __init__(
        self, eos_threshold: float = -4.0, temperature: float | None = None
    ) -> None:
        self._model: TTSModel | None = None
        self._eos_threshold = eos_threshold
        self._temperature = temperature

    @property
    def model(self) -> TTSModel:
        if self._model is None:
            self._model = TTSModel.load_model(
                eos_threshold=self._eos_threshold, temp=self._temperature
            )
        return self._model

    def list_saved_voices(self) -> list[str]:
        """Return the names of voices that have been cloned and saved locally."""
        return sorted(p.stem for p in VOICES_DIR.glob("*.safetensors"))

    def delete_saved_voice(self, voice_name: str) -> None:
        path = VOICES_DIR / f"{voice_name}.safetensors"
        path.unlink(missing_ok=True)

    def clone_voice_from_audio(self, audio_bytes: bytes, voice_name: str) -> Path:
        """
        Clone a voice from an uploaded audio sample (wav/mp3/m4a/flac/ogg
        bytes) and save it under `voice_name` for reuse.
        """
        tmp_path = VOICES_DIR / f"_tmp_{voice_name}.wav"
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            sf.write(tmp_path, audio_data, sample_rate)
            state = self.model.get_state_for_audio_prompt(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        save_path = VOICES_DIR / f"{voice_name}.safetensors"
        export_model_state(state, str(save_path))
        return save_path

    def _load_voice_state(self, voice_name: str) -> dict:
        saved_path = VOICES_DIR / f"{voice_name}.safetensors"
        if saved_path.exists():
            return self.model.get_state_for_audio_prompt(str(saved_path))
        if voice_name in BUILT_IN_VOICES:
            return self.model.get_state_for_audio_prompt(BUILT_IN_VOICES[voice_name])
        return self.model.get_state_for_audio_prompt(voice_name)

    def generate(self, text: str, voice_name: str):
        """
        Generate speech audio for `text` using `voice_name`.
        Returns (audio_samples: np.ndarray, sample_rate: int).
        """
        voice_state = self._load_voice_state(voice_name)
        audio_tensor: torch.Tensor = self.model.generate_audio(voice_state, text)
        audio = audio_tensor.detach().cpu().numpy()
        return audio, self.model.sample_rate

    def generate_by_sentence(self, text: str, voice_name: str):
        """
        Generate speech sentence-by-sentence and stitch the results
        together with a short silence between them.

        This is more robust than generating the whole text in one call:
        if the model stumbles on one sentence, it doesn't destabilize
        the ones around it. Returns (audio_samples, sample_rate, warnings)
        where `warnings` lists any sentences that failed outright.
        """
        sentences = _split_into_sentences(text)
        if not sentences:
            return self.generate(text, voice_name)[0], self.model.sample_rate, []

        voice_state_template = self._load_voice_state(voice_name)
        sample_rate = self.model.sample_rate
        silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)

        chunks: list[np.ndarray] = []
        warnings: list[str] = []
        for sentence in sentences:
            try:
                audio_tensor = self.model.generate_audio(voice_state_template, sentence)
                chunks.append(audio_tensor.detach().cpu().numpy())
                chunks.append(silence)
            except Exception:
                warnings.append(sentence)

        if not chunks:
            raise RuntimeError("Every sentence failed to generate.")

        return np.concatenate(chunks), sample_rate, warnings


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter — splits on ./!/? followed by whitespace."""
    text = text.strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in pieces if p.strip()]
