#!/usr/bin/env python3
# /// script
# dependencies = [
#   "fastcore",
#   "faster-whisper",
#   "sounddevice", 
#   "pyperclip",
#   "numpy"
# ]
# ///
from faster_whisper import WhisperModel
from pathlib import Path

import numpy as np, sounddevice as sd
import argparse,os,pyperclip,signal,sys,tempfile

pid_file = Path(tempfile.gettempdir()) / "speech-buddy.pid"
audio_chunks = []
recording = False

def signal_handler(signum, frame):
    global recording
    recording = False

def audio_callback(indata, frames, time, status):
    if recording: audio_chunks.append(indata.copy())

def start():
    global recording
    
    pid_file.write_text(str(os.getpid()))
    signal.signal(signal.SIGUSR1, signal_handler)
    
    print("Recording started...")
    recording = True
    
    with sd.InputStream(callback=audio_callback, samplerate=16000, channels=1):
        while recording: sd.sleep(100)
    
    print("Processing...")
    audio_flat = np.concatenate(audio_chunks).flatten().astype(np.float32)
    model = WhisperModel("base", device="cpu")
    segments, _ = model.transcribe(audio_flat)
    text = " ".join([seg.text for seg in segments]).strip()
    
    pyperclip.copy(text)
    print(f"Copied: {text}")
    
    pid_file.unlink(missing_ok=True)

def stop():
    if not pid_file.exists(): return print("No active recording.")
    
    pid = int(pid_file.read_text())
    os.kill(pid, signal.SIGUSR1)
    print("Stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"])
    args = parser.parse_args()
    
    start() if args.action == "start" else stop()
