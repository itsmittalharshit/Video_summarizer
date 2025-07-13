!pip install moviepy opencv-python faster-whisper transformers torch torchvision torchaudio
!pip install yt-dlp
import os
import numpy as np
import streamlit as st
from moviepy import VideoFileClip
from faster_whisper import WhisperModel
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import tempfile
import yt_dlp
import torch

device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to("cpu")
t5_summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

def get_temp_path(filename):
    return os.path.join(tempfile.mkdtemp(), filename)


def download_video_from_url(url, save_path):
    ydl_opts = {
        'outtmpl': save_path,
        'format': 'best[ext=mp4]/bestvideo+bestaudio/best',
        'quiet': True,
        'merge_output_format': 'mp4',
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info).replace('.webm', '.mp4')

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, logger=None)
    return audio_path

def transcribe_audio(audio_path):
    model = WhisperModel("base.en", device="cuda" if torch.cuda.is_available() else "cpu",
                         compute_type="float16" if torch.cuda.is_available() else "int8")
    segments, _ = model.transcribe(audio_path)
    return " ".join([seg.text for seg in segments])


def summarize_text(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = []
    for chunk in chunks:
        input_len = len(chunk.split())
        max_len = max(30, min(100, input_len // 3))
        summary = t5_summarizer(chunk, max_length=max_len, min_length=15, do_sample=False)[0]["summary_text"]
        summaries.append(summary)
    return " ".join(summaries)


st.set_page_config(page_title="🎬 Video Summarizer", layout="wide")
st.title("🎬 AI Video Summarizer")

video_source = st.radio("Choose video input method:", ["YouTube URL", "Upload Video File"])

video_path = None
if video_source == "YouTube URL":
    video_url = st.text_input("Paste YouTube video URL:")
    if st.button("Submit") and video_url:
        with st.spinner("⏳ Processing your video... please wait"):
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "video.mp4")
            try:
                video_path = download_video_from_url(video_url, output_path)
            except Exception as e:
                st.error(f"❌ Failed to download video: {e}")
                st.stop()

elif video_source == "Upload Video File":
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None and st.button("Submit"):
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
            

if video_path:
    with st.spinner("Wait...Summary is being converted"):
        audio_path = get_temp_path("audio.wav")
        extract_audio(video_path, audio_path)
        transcript = transcribe_audio(audio_path)
        summary = summarize_text(transcript)

    st.video(video_path)
    st.markdown("### 📝 Summary", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background-color: #f9f9f9; border-left: 6px solid #4CAF50; padding: 15px; font-size: 16px; line-height: 1.6; border-radius: 5px;">
        {summary}
        </div>
        """,
        unsafe_allow_html=True
    )


