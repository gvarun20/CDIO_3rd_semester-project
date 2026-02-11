# audio_processor.py
import numpy as np
import librosa
import librosa.display
import tempfile
from scipy.io import wavfile
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
import pyaudio
import os
import mido


class AudioProcessor:
    def __init__(self, sample_rate=40960):
        self.sample_rate = sample_rate
        self.chunk_size = 2048

    def load_audio(self, file_path):
        """Load audio file using librosa"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            duration = len(y) / sr
            return y, sr, duration
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")

    def compute_spectrogram(self, audio_data, sr):
        """Compute spectrogram using STFT"""
        stft = librosa.stft(audio_data)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        return stft, spectrogram

    def compute_waveform(self, audio_data):
        """Compute waveform data"""
        times = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
        return times, audio_data


class OnsetOffsetDetector:
    """
    Detect onsets and approximate offsets from an audio signal.
    Uses librosa's onset strength envelope + tuned peak picking,
    then derives offsets from the next onset or the end of the file.
    """

    def __init__(
        self,
        sample_rate=40960,
        hop_length=1024,
        backtrack=True,
        min_gap=0.05,        # seconds: merge onsets closer than this
        min_note_duration=0.08 # seconds: enforce minimum duration
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.backtrack = backtrack
        self.min_gap = min_gap
        self.min_note_duration = min_note_duration

    def detect_onset_offset(self, audio_data, sr):
        """
        Parameters
        ----------
        audio_data : np.ndarray
            Mono audio signal.
        sr : int
            Sample rate.

        Returns
        -------
        onsets : np.ndarray
            Onset times in seconds.
        offsets : np.ndarray
            Offset times in seconds (approx., next onset or min duration or end of audio).
        """
        # --- 1) Compute onset strength envelope ---
        onset_env = librosa.onset.onset_strength(
            y=audio_data,
            sr=sr,
            hop_length=self.hop_length
        )

        # Optional: simple smoothing of the envelope (moving average)
        if len(onset_env) >= 3:
            kernel = np.ones(3) / 3.0
            onset_env = np.convolve(onset_env, kernel, mode="same")

        # --- 2) Peak picking on the envelope ---
        # These parameters can be tweaked if needed:
        onset_times = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=self.backtrack,
            units="time",
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=0.06,
            wait=1
        )

        
        if onset_times.size == 0:
            return np.array([]), np.array([])

        # --- 3) Merge onsets that are too close (min_gap) ---
        merged_onsets = [float(onset_times[0])]
        for t in onset_times[1:]:
            if (t - merged_onsets[-1]) >= self.min_gap:
                merged_onsets.append(float(t))
            # else: t is too close; treat as the same onset

        onsets = np.array(merged_onsets, dtype=float)

        # --- 4) Derive offsets from next onset or audio end, enforce min duration ---
        audio_duration = len(audio_data) / float(sr)
        offsets = []

        for i, onset in enumerate(onsets):
            if i + 1 < len(onsets):
                # Next onset
                candidate_offset = onsets[i + 1]
            else:
                # Last note: candidate is end of audio
                candidate_offset = audio_duration

            # Enforce minimum note duration
            min_allowed = onset + self.min_note_duration
            offset = max(candidate_offset, min_allowed)

            # But never exceed the actual audio duration
            offset = min(offset, audio_duration)

            offsets.append(offset)

        return onsets, np.array(offsets, dtype=float)


class NoteMapping:
    """
    Map detected fundamental frequencies (f0) to musical notes.
    Assumes equal-tempered tuning with A4 = 440 Hz.
    """

    def __init__(self, f_ref=440.0, midi_ref=69,
                 midi_min=36, midi_max=96):
        self.f_ref = f_ref
        self.midi_ref = midi_ref
        self.midi_min = midi_min
        self.midi_max = midi_max

        # Precompute name table for MIDI notes
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                           'F#', 'G', 'G#', 'A', 'A#', 'B']

    def freq_to_midi(self, f0):
        """
        Map a frequency in Hz to the nearest MIDI note number.
        Returns None if f0 is non-positive or outside supported range.
        """
        if f0 is None or f0 <= 0:
            return None

        midi = self.midi_ref + 12 * np.log2(f0 / self.f_ref)
        midi_rounded = int(round(midi))

        if midi_rounded < self.midi_min or midi_rounded > self.midi_max:
            return None

        return midi_rounded

    def midi_to_name(self, midi):
        """
        Convert a MIDI note number to (note_name, octave), e.g. ("C#", 4).
        """
        if midi is None:
            return None, None

        note_index = midi % 12
        octave = (midi // 12) - 1   # MIDI definition
        return self.note_names[note_index], octave

    def map_single_frequency(self, f0):
        """
        Convenience method: f0 -> (midi, note_name, octave).
        """
        midi = self.freq_to_midi(f0)
        if midi is None:
            return None, None, None
        name, octave = self.midi_to_name(midi)
        return midi, name, octave


class EnhancedPitchDetector:
    """
    Pitch detection and mapping to musical note names.
    Uses different methods for monophonic vs. polyphonic detection.
    """
    def __init__(self, sample_rate=40960, fmin=None, fmax=None):
        self.sample_rate = sample_rate
        # Reasonable range for a 61-key keyboard (C2–C7)
        if fmin is None:
            fmin = librosa.note_to_hz("C2")  # ~65 Hz
        if fmax is None:
            fmax = librosa.note_to_hz("C7")  # ~2093 Hz
        self.fmin = fmin
        self.fmax = fmax

        self.note_mapper = NoteMapping(midi_min=36, midi_max=96)

    def detect_multiple_pitches(self, audio_segment, max_notes=2):
        """
        Detect one (monophonic) or more (polyphonic) pitches in a note segment.
        Uses pYIN for monophonic and CQT peak analysis for polyphonic.
        
        Parameters
        ----------
        audio_segment : np.ndarray
            The audio segment of a single note event.
        max_notes : int
            The maximum number of simultaneous notes to detect.

        Returns
        -------
        list of float
            A list of detected fundamental frequencies (in Hz).
        """
        if audio_segment is None or len(audio_segment) == 0:
            return []

        # Only analyze the attack/transient portion (first 0.5 seconds)
        max_duration = 0.5  # seconds
        max_samples = int(max_duration * self.sample_rate)
        analysis_segment = audio_segment[:max_samples]

        if len(analysis_segment) == 0:
            return []

        # Skip very low-energy segments
        rms = np.sqrt(np.mean(analysis_segment**2))
        if rms < 0.005:
            return []

        if max_notes == 1:
            # --- Monophonic Mode (pYIN) ---
            f0, voiced_flag, voiced_prob = librosa.pyin(
                analysis_segment,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.sample_rate,
                frame_length=1024,
                hop_length=256
            )
            
            # Filter by voicing probability (> 0.8) and remove unvoiced frames (NaNs)
            f0 = f0[~np.isnan(f0) & (voiced_prob > 0.8)]
            
            if len(f0) == 0:
                return []

            # Dominant pitch = median f0
            main_f0 = float(np.median(f0))
            return [main_f0]
        else:
            # --- Polyphonic Mode (CQT Spectral Peak Detection) ---
            
            # 1. Compute CQT
            C = librosa.cqt(
                analysis_segment,
                sr=self.sample_rate,
                hop_length=512,
                fmin=self.fmin,
                n_bins=84,  # 7 octaves coverage
                bins_per_octave=12
            )
            
            C_mag = np.abs(C)
            
            # 2. Sum magnitudes across time frames
            C_sum = np.sum(C_mag, axis=1) 
            
            # 3. Find indices of the top N bins
            top_indices = np.argpartition(C_sum, -max_notes)[-max_notes:]
            top_indices = top_indices[np.argsort(-C_sum[top_indices])]
            
            pitches_hz = []
            
            # 4. Get the exact CQT frequencies (FIX: removed 'sr' arg)
            cqt_freqs = librosa.cqt_frequencies(
                n_bins=84, 
                fmin=self.fmin, 
                bins_per_octave=12
            )
            
            # 5. Filter and map to note centers
            threshold = np.max(C_sum) * 0.1 
            
            for index in top_indices:
                if C_sum[index] > threshold:
                    freq = cqt_freqs[index]
                    
                    # Map the CQT frequency to the nearest MIDI note center frequency 
                    midi = self.note_mapper.freq_to_midi(freq)
                    if midi is not None:
                        # Convert MIDI back to its precise center frequency
                        center_freq = librosa.midi_to_hz(midi)
                        pitches_hz.append(center_freq)
            
            # Remove duplicate note detections and limit to max_notes
            unique_pitches = list(set(pitches_hz))[:max_notes]
            return unique_pitches

    def pitch_to_note(self, pitch_hz):
        """
        Convert a pitch in Hz to a note string like 'C#4'
        using the NoteMapping class.
        """
        if pitch_hz is None:
            return None

        midi, name, octave = self.note_mapper.map_single_frequency(pitch_hz)
        if name is None:
            return None

        return f"{name}{octave}"

class EvaluationEngine:
    """
    Calculates music transcription performance metrics (Precision, Recall, F-score)
    by comparing estimated note events against ground truth note events.
    """
    def __init__(self):
        self.note_mapper = NoteMapping()

    def calculate_metrics(self, estimated_results, ground_truth_events):
        """
        Calculates Note Detection metrics (Precision, Recall, F-score).
        
        Parameters
        ----------
        estimated_results : list
            Output from TranscriptionEngine.
        ground_truth_events : list
            List of note events derived from the GT MIDI file.
            (Format: [{'onset': t, 'offset': t, 'note': 'C4', 'midi': 60}, ...])

        Returns
        -------
        dict
            Performance metrics including TP, FP, FN, Precision, Recall, F-score.
        """

        # --- 1. Flatten estimated results into a list of single note objects ---
        est_notes = []
        for event in estimated_results:
            # Added robustness checks for structure
            if 'notes' in event and isinstance(event['notes'], list):
                for note_data in event['notes']:
                    if 'pitch' in note_data:
                        midi = self.note_mapper.freq_to_midi(note_data['pitch'])
                        if midi is not None:
                            est_notes.append({
                                'onset': event['onset'],
                                'offset': event['offset'],
                                'midi': midi,
                                'matched': False # <-- Ensure Estimated notes are initialized
                            })
        
        # --- 2. Prepare Ground Truth (GT) and match lists (FIXED LOGIC) ---
        gt_notes = []
        for n in ground_truth_events:
            # Create a copy and EXPLICITLY initialize the 'matched' flag
            gt_note = n.copy()
            gt_note['matched'] = False 
            gt_notes.append(gt_note)

        TP = 0
        
        # --- 3. CORE MATCHING LOGIC ---
        # Match each estimated note to the best available ground truth note
        for est_note in est_notes:
            for gt_note in gt_notes:
                
                # The error was likely raised here because 'matched' was not in gt_note
                if gt_note['matched']: 
                    continue

                # 3a. Pitch Check (Match MIDI number)
                if est_note['midi'] != gt_note['midi']:
                    continue

                # 3b. Timing Check (Temporal Overlap)
                
                overlap_start = max(est_note['onset'], gt_note['onset'])
                overlap_end = min(est_note['offset'], gt_note['offset'])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                gt_duration = gt_note['offset'] - gt_note['onset']
                est_duration = est_note['offset'] - est_note['onset']
                
                # Handle cases where duration might be near zero (safety check)
                if gt_duration <= 0 or est_duration <= 0:
                    continue

                min_duration = min(gt_duration, est_duration)
                
                if overlap_duration > (0.5 * min_duration):
                    # Found a valid True Positive (TP) match
                    gt_note['matched'] = True
                    est_note['matched'] = True
                    TP += 1
                    break # Move to the next estimated note
        
        # --- 4. Calculate FP and FN ---
        # False Negative: GT notes that were not matched (missed by the system)
        FN = sum(1 for gt in gt_notes if not gt['matched'])
        
        # False Positive: Estimated notes that were not matched (hallucinated by the system)
        FP = sum(1 for est in est_notes if not est['matched'])
        
        # --- 5. Calculate Metrics (Added division by zero checks) ---
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        # F-score (harmonic mean)
        f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'TP': TP, 'FP': FP, 'FN': FN,
            'Precision': precision, 
            'Recall': recall,
            'F-score': f_score
        }







class TranscriptionEngine:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.onset_detector = OnsetOffsetDetector()
        self.pitch_detector = EnhancedPitchDetector()
        self.results = []

    def _postprocess_segments(self, segments,
                              min_duration=0.08,    # 80 ms
                              merge_gap=0.05):      # 50 ms
        """
        Clean up transcription segments:
        - remove very short blips
        - merge consecutive segments with same notes and tiny gap
        """
        cleaned = []

        for evt in segments:
            # 1) Skip very short notes (likely noise or onset glitch)
            if evt['duration'] < min_duration:
                continue

            # Convert notes list to a tuple of note strings for comparison
            current_notes_tuple = tuple(sorted([n['note'] for n in evt['notes']]))

            if cleaned:
                last = cleaned[-1]
                last_notes_tuple = tuple(sorted([n['note'] for n in last['notes']]))
                
                # Check for same notes (handles chords too)
                same_notes = current_notes_tuple == last_notes_tuple
                small_gap = evt['onset'] - last['offset'] < merge_gap

                # 2) Merge with previous if same note(s) and very close
                if same_notes and small_gap:
                    last['offset'] = evt['offset']
                    last['duration'] = last['offset'] - last['onset']
                    # Re-calculate average pitch/frequency if needed, but for simplicity we keep the first
                    continue

            # Otherwise, keep as new event
            cleaned.append(evt)

        return cleaned


    def transcribe_audio(self, audio_data, sr, max_simultaneous_notes=2):
        """Enhanced transcription pipeline with multi-pitch support"""
        try:
            # Detect onsets and offsets
            onsets, offsets = self.onset_detector.detect_onset_offset(audio_data, sr)

            transcription = []

            # Process each note segment
            for i, onset in enumerate(onsets):
                onset_sample = int(onset * sr)

                # Find corresponding offset
                offset = offsets[i] if i < len(offsets) else len(audio_data) / sr
                offset_sample = int(offset * sr)

                # Extract note segment
                note_segment = audio_data[onset_sample:offset_sample]

                if len(note_segment) > 0:
                    # Detect multiple pitches (or single pitch if max_notes=1)
                    pitches = self.pitch_detector.detect_multiple_pitches(
                        note_segment,
                        max_notes=max_simultaneous_notes
                    )

                    notes = []
                    for pitch in pitches:
                        note_name = self.pitch_detector.pitch_to_note(pitch)
                        if note_name:
                            notes.append({
                                'pitch': pitch,
                                'note': note_name
                            })

                    if notes:
                        transcription.append({
                            'onset': onset,
                            'offset': offset,
                            'notes': notes,
                            'duration': offset - onset,
                            'note_count': len(notes)
                        })
            transcription = self._postprocess_segments(transcription)            
            return transcription
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")


class RecordingManager:
    def __init__(self):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.is_recording = False
        self.audio_stream = None
        self.recording_frames = []

    def start_recording(self, sample_rate=40960, chunk_size=2048):
        """Start audio recording"""
        try:
            self.is_recording = True
            self.recording_frames = []

            # Configure audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
                stream_callback=self.audio_callback
            )
            self.audio_stream.start_stream() 
            return True
        except Exception as e:
            raise Exception(f"Failed to start recording: {str(e)}. (Do you have a microphone connected?)")

    def stop_recording(self):
        """Stop audio recording"""
        if self.is_recording:
            self.is_recording = False
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time recording"""
        if self.is_recording:
            self.recording_frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def get_recorded_audio(self, sample_rate=40960):
        """Convert recorded frames to audio data"""
        if not self.recording_frames:
            return None, sample_rate

        audio_data = np.frombuffer(b''.join(self.recording_frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize

        return audio_data, sample_rate

    def save_recording(self, file_path, sample_rate=40960):
        """Save recorded audio to WAV file using scipy.io.wavfile"""
        audio_data, sr = self.get_recorded_audio(sample_rate)
        if audio_data is not None:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(file_path, sr, audio_int16)
            return file_path
        return None

    def cleanup(self):
        """Clean up PyAudio resources"""
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()


# -------- Utility functions (GLOBAL) --------

def save_audio_data(file_path, audio_data, sample_rate):
    """Save audio data to WAV file"""
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(file_path, sample_rate, audio_int16)


def load_audio_file(file_path, target_sr=None):
    """
    Load an audio file using librosa (WAV/MP3/etc).
    If a MIDI file is passed, raise a clear error so the caller
    can route it to a MIDI-specific loader.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".mid", ".midi"]:
        raise Exception(
            f"'{file_path}' is a MIDI file, not audio. "
            "Use the MIDI loader for this format."
        )

    try:
        audio_data, sr = librosa.load(file_path, sr=target_sr)
        duration = len(audio_data) / sr
        return audio_data, sr, duration
    except Exception as e:
        raise Exception(f"Error loading audio file {file_path}: {str(e)}")


def load_midi_file(file_path):
    """
    Parse a MIDI file and return a list of note events
    with onset, offset (in seconds) and note names.
    """
    mid = mido.MidiFile(file_path)
    current_time = 0.0  # seconds

    # Active notes: midi -> onset_time_in_seconds
    active_notes = {}
    mapper = NoteMapping()

    events = []

    for msg in mid:
        current_time += msg.time

        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = current_time

        elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active_notes:
                onset = active_notes.pop(msg.note)
                offset = current_time
                duration = offset - onset

                name, octave = mapper.midi_to_name(msg.note)
                note_label = f"{name}{octave}" if name is not None else None

                events.append({
                    "onset": onset,
                    "offset": offset,
                    "duration": duration,
                    "midi": msg.note,
                    "note": note_label
                })

    return events

def transcribe_midi_file(file_path):
    """Transcribe a MIDI file into note events."""
    try:
        events = load_midi_file(file_path)
        return events
    except Exception as e:
        raise Exception(f"Error transcribing MIDI file {file_path}: {str(e)}")
def transcribe_audio_file(file_path, target_sr=40960, max_simultaneous_notes=2):
    """ Transcribe an audio file into note events."""
    try:
        audio_data, sr, duration = load_audio_file(file_path, target_sr=target_sr)
        engine = TranscriptionEngine()
        transcription = engine.transcribe_audio(
            audio_data,
            sr,
            max_simultaneous_notes=max_simultaneous_notes
        )
        return transcription
    except Exception as e:
        raise Exception(f"Error transcribing audio file {file_path}: {str(e)}")