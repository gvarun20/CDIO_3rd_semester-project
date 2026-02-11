# 🎵 Music Transcription System

A sophisticated desktop application for automatic music transcription that converts audio recordings into musical notation with support for polyphonic detection.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

The Music Transcription System is a comprehensive tool that combines signal processing, machine learning principles, and music theory to automatically transcribe audio recordings into musical notation. It supports both monophonic (single-note) and polyphonic (multi-note) detection, making it suitable for various musical instruments including pianos, keyboards, and other melodic instruments.

## ✨ Key Features

### 🎹 Advanced Transcription Capabilities
- **Multi-Pitch Detection**: Simultaneously detect up to 3 notes (configurable)
- **Real-time Processing**: Live audio recording and immediate transcription
- **Polyphonic Support**: Accurate chord and multi-instrument detection
- **Precise Timing**: Note onset/offset detection with millisecond accuracy

### 📊 Visualization & Analysis
- **Waveform Display**: Real-time audio waveform visualization
- **Spectrogram Analysis**: Frequency domain analysis with interactive plots
- **Multi-track Visualization**: Separate display of different frequency components

### 💾 Data Management
- **SQLite Database**: Secure storage of transcription history and results
- **Export Functionality**: Export results to text files for external use
- **Session Management**: Complete history of all processing sessions

### 🎚️ Flexible Input Options
- **File Processing**: Support for WAV, MP3, FLAC, M4A, AAC formats
- **Live Recording**: Direct microphone input with real-time processing
- **Adjustable Settings**: Configurable sample rates and detection sensitivity

## 🛠️ Technical Architecture

### Core Components

#### 1. Audio Processing Engine (`audio_processor.py`)
```python
# Advanced multi-pitch detection algorithm
def detect_multiple_pitches(self, audio_chunk, max_notes=2):
    # Spectral peak detection + harmonic grouping
    # Musicological filtering for plausible note combinations