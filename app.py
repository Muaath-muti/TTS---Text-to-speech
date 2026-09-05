"""
University Text-to-Speech tool, built on Kyutai's Pocket TTS.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path

import streamlit as st
from PIL import Image

from audio_utils import to_mp3_bytes, to_wav_bytes
from tts_engine import BUILT_IN_VOICES, TTSEngine

LOGO_PATH = Path(__file__).parent / "assets" / "uwc_logo.png"

st.set_page_config(
    page_title="Voice of Good Hope — UWC",
    page_icon=Image.open(LOGO_PATH) if LOGO_PATH.exists() else "🗣️",
    layout="wide",
)

# --------------------------------------------------------------------
# Styling — University of the Western Cape colors: royal blue, gold, white
# --------------------------------------------------------------------
_logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    .block-container {{
        max-width: 1100px;
        margin: 0 auto;
        padding-top: 2.2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }}

    h1, h2, h3 {{ font-family: 'Fraunces', serif !important; font-weight: 600 !important; color: #010C80; }}

    .app-header {{
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.2rem;
    }}
    .app-header img {{ height: 64px; width: 64px; border-radius: 6px; }}
    .app-header .titles h1 {{ margin: 0; font-size: 1.6rem; line-height: 1.15; }}
    .app-header .titles .subtitle {{
        color: #4A4F7A;
        font-size: 1rem;
        margin-top: 0.1rem;
    }}

    .accent-bar {{
        height: 5px;
        border-radius: 3px;
        margin: 0.6rem 0 1.6rem 0;
        background: linear-gradient(90deg, #010C80 0%, #010C80 45%, #B9AB60 45%, #B9AB60 80%, #FFFFFF 80%, #FFFFFF 100%);
        border: 1px solid #E4E1D0;
    }}

    .stTabs [data-baseweb="tab"] {{ font-weight: 500; font-size: 1rem; }}
    .stTabs [aria-selected="true"] {{ color: #010C80 !important; border-bottom-color: #B9AB60 !important; }}

    div.stButton > button, div.stDownloadButton > button {{
        border-radius: 8px;
        border: 1px solid #010C80;
        background-color: #010C80;
        color: white;
        font-weight: 500;
    }}
    div.stButton > button:hover, div.stDownloadButton > button:hover {{
        background-color: #B9AB60;
        border-color: #B9AB60;
        color: #010C80;
    }}

    .voice-row {{
        padding: 0.6rem 0.9rem;
        border: 1px solid #D9D3B4;
        border-left: 4px solid #B9AB60;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        background-color: #FAFAF6;
        color: #1A1D3A;
        font-weight: 500;
    }}

    section[data-testid="stSidebar"] {{ background-color: #010C80; }}
    section[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
    section[data-testid="stSidebar"] .stSlider [role="slider"] {{ background-color: #B9AB60 !important; }}
    section[data-testid="stSidebar"] .stSlider > div > div > div > div {{ background-color: #B9AB60 !important; }}

    .meta-line {{ color: #8A8A85; font-size: 0.85rem; }}
    </style>

    <div class="app-header">
        <img src="data:image/png;base64,{_logo_b64}" />
        <div class="titles">
            <h1>Voice of Good Hope</h1>
            <div class="subtitle">Text-to-speech for UWC, powered by Kyutai's Pocket TTS.</div>
        </div>
    </div>
    <div class="accent-bar"></div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# Engine (with stability controls in the sidebar)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model (first run only)...")
def get_engine(eos_threshold: float, temperature: float | None) -> TTSEngine:
    return TTSEngine(eos_threshold=eos_threshold, temperature=temperature)


with st.sidebar:
    st.header("Settings")
    st.caption(
        "If a voice breaks up or repeats on longer text, raise the "
        "sensitivity below and regenerate. Changing this reloads the model."
    )
    eos_threshold = st.slider(
        "End-of-sentence sensitivity", min_value=-8.0, max_value=-1.0, value=-4.0, step=0.5
    )
    use_custom_temp = st.checkbox("Override temperature (advanced)", value=False)
    temperature = (
        st.slider("Temperature", min_value=0.1, max_value=1.2, value=0.7, step=0.1)
        if use_custom_temp
        else None
    )

engine = get_engine(eos_threshold, temperature)

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: text, voice, audio, sr, time

SAMPLE_TEXT = (
    "Welcome to Voice of Good Hope, the text-to-speech tool built for the "
    "University of the Western Cape. Type any text here, choose a voice, "
    "and press generate to hear it read aloud."
)
if "text_input_value" not in st.session_state:
    st.session_state.text_input_value = SAMPLE_TEXT

tab_generate, tab_library, tab_add_voice = st.tabs(
    ["Generate speech", "Voice library", "Add a new voice"]
)

# ----------------------------------------------------------------------
# TAB 1: Generate speech
# ----------------------------------------------------------------------
with tab_generate:
    saved_voices = engine.list_saved_voices()
    all_voice_labels = list(BUILT_IN_VOICES.keys()) + saved_voices

    text = st.text_area(
        "Text to speak",
        placeholder="Type or paste the text you want read aloud...",
        height=160,
        key="text_input_value",
    )
    st.caption("Sample text loaded above — press \"Generate audio\" to hear it, or replace it with your own.")

    word_count = len(text.split())
    est_seconds = max(word_count / 2.5, 0)  # rough speaking-rate estimate
    st.caption(f"{word_count} words · roughly {est_seconds:.0f}s of audio")

    voice_label = st.selectbox("Voice", options=all_voice_labels)
    output_formats = st.multiselect(
        "Download as", options=["wav", "mp3"], default=["mp3"]
    )
    split_sentences = st.checkbox(
        "Generate sentence-by-sentence (recommended for longer text)",
        value=True,
        help="Generates each sentence separately and stitches them together. "
        "More reliable for multi-sentence text — if one sentence struggles, "
        "it won't drag the others down with it.",
    )

    if st.button("Generate audio", type="primary", disabled=not text.strip()):
        voice_id = BUILT_IN_VOICES.get(voice_label, voice_label)
        try:
            with st.spinner("Generating..."):
                if split_sentences:
                    audio, sample_rate, failed = engine.generate_by_sentence(text, voice_id)
                else:
                    audio, sample_rate = engine.generate(text, voice_id)
                    failed = []

            st.session_state.history.insert(
                0,
                {
                    "text": text,
                    "voice": voice_label,
                    "audio": audio,
                    "sr": sample_rate,
                    "time": dt.datetime.now().strftime("%H:%M:%S"),
                },
            )
            st.session_state.history = st.session_state.history[:10]

            if failed:
                st.warning(
                    "Most of the text generated fine, but "
                    f"{len(failed)} sentence(s) couldn't be generated and were skipped: "
                    + " / ".join(f'"{s}"' for s in failed)
                )

            st.audio(audio, sample_rate=sample_rate)

            cols = st.columns(len(output_formats)) if output_formats else []
            for col, fmt in zip(cols, output_formats):
                with col:
                    try:
                        if fmt == "wav":
                            st.download_button(
                                "Download .wav",
                                data=to_wav_bytes(audio, sample_rate),
                                file_name="speech.wav",
                                mime="audio/wav",
                            )
                        else:
                            st.download_button(
                                "Download .mp3",
                                data=to_mp3_bytes(audio, sample_rate),
                                file_name="speech.mp3",
                                mime="audio/mpeg",
                            )
                    except Exception as e:
                        st.error(f"Couldn't prepare the {fmt} download.\n\nDetails: {e}")
        except Exception as e:
            st.error(f"Couldn't generate that audio.\n\nDetails: {e}")

    if st.session_state.history:
        st.divider()
        st.subheader("Recent generations")
        for i, item in enumerate(st.session_state.history):
            with st.expander(f"{item['time']} · {item['voice']} · \"{item['text'][:50]}...\""):
                try:
                    st.audio(item["audio"], sample_rate=item["sr"])
                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.download_button(
                            "Download .wav",
                            data=to_wav_bytes(item["audio"], item["sr"]),
                            file_name=f"speech_{i}.wav",
                            mime="audio/wav",
                            key=f"hist_wav_{i}",
                        )
                    with dcol2:
                        st.download_button(
                            "Download .mp3",
                            data=to_mp3_bytes(item["audio"], item["sr"]),
                            file_name=f"speech_{i}.mp3",
                            mime="audio/mpeg",
                            key=f"hist_mp3_{i}",
                        )
                except Exception as e:
                    st.error(f"Couldn't load this item.\n\nDetails: {e}")

# ----------------------------------------------------------------------
# TAB 2: Voice library
# ----------------------------------------------------------------------
with tab_library:
    st.write("Voices available to everyone using this app.")

    st.markdown("**Built-in voices**")
    for label in BUILT_IN_VOICES:
        st.markdown(f'<div class="voice-row">{label}</div>', unsafe_allow_html=True)

    saved_voices = engine.list_saved_voices()
    st.markdown("**Cloned voices**")
    if not saved_voices:
        st.caption("No cloned voices yet — add one in the \"Add a new voice\" tab.")
    for voice_name in saved_voices:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(f'<div class="voice-row">{voice_name}</div>', unsafe_allow_html=True)
        with c2:
            if st.button("Delete", key=f"del_{voice_name}"):
                engine.delete_saved_voice(voice_name)
                st.rerun()

# ----------------------------------------------------------------------
# TAB 3: Add a new voice (voice cloning)
# ----------------------------------------------------------------------
with tab_add_voice:
    st.write(
        "Provide a clean audio sample (10-30 seconds, one speaker, "
        "minimal background noise) to clone a new voice — for example, "
        "a South African-accented English speaker. You can either "
        "upload a file or record directly in your browser."
    )

    new_voice_name = st.text_input(
        "Name this voice",
        placeholder="e.g. thabo_sa_english",
        help="Used as the internal file name — letters, numbers, and underscores only.",
    )

    source_choice = st.radio(
        "Voice sample source",
        options=["Upload a file", "Record with microphone"],
        horizontal=True,
    )

    voice_sample_bytes: bytes | None = None

    if source_choice == "Upload a file":
        uploaded_audio = st.file_uploader(
            "Voice sample", type=["wav", "mp3", "m4a", "flac", "ogg"]
        )
        if uploaded_audio is not None:
            voice_sample_bytes = uploaded_audio.read()
    else:
        st.caption(
            "Click record, speak clearly for 10-30 seconds, then click stop. "
            "Your browser will ask permission to use the microphone the first time."
        )
        recorded_audio = st.audio_input("Record your voice sample")
        if recorded_audio is not None:
            voice_sample_bytes = recorded_audio.read()
            st.audio(recorded_audio)

    can_clone = bool(new_voice_name.strip()) and voice_sample_bytes is not None
    if st.button("Clone and save this voice", disabled=not can_clone):
        try:
            with st.spinner("Cloning voice... this can take a little while on CPU."):
                engine.clone_voice_from_audio(voice_sample_bytes, new_voice_name.strip())
            st.success(
                f"Saved as '{new_voice_name.strip()}'. Taking you to the voice list..."
            )
            st.rerun()
        except Exception as e:
            st.error(
                "Couldn't process that audio sample. Try a different file or re-record it.\n\n"
                f"Details: {e}"
            )
