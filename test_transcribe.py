import os
from main import TranscriptionEngine, load_audio_file, load_midi_file

if __name__ == "__main__":
    # Change this line depending on what you want to test
    audio_path = "EHH2_90_Cm_vinyl_piano_08_2BAR_piano_90BPM_Cminor_BANDLAB.wav"  

    ext = os.path.splitext(audio_path)[1].lower()

    # ----------------------------------------------------
    # 1) MIDI FILE: use symbolic parsing (no audio)
    # ----------------------------------------------------
    if ext in [".mid", ".midi"]:
        try:
            events = load_midi_file(audio_path)
            print(f"Loaded MIDI file: {audio_path}")
            if not events:
                print("No MIDI note events found.")
            else:
                for evt in events:
                    print(
                        f"Onset: {evt['onset']:.3f}s | "
                        f"Offset: {evt['offset']:.3f}s | "
                        f"Duration: {evt['duration']:.3f}s | "
                        f"MIDI: {evt['midi']} | "
                        f"Note: {evt['note']}"
                    )
        except Exception as e:
            print(f"Error while parsing MIDI file: {e}")

    # ----------------------------------------------------
    # 2) AUDIO FILE: WAV/MP3/etc → use TranscriptionEngine
    # ----------------------------------------------------
    else:
        try:
            audio_data, sr, duration = load_audio_file(audio_path, target_sr=40960)
            print(f"Loaded {audio_path} (sr={sr}, duration={duration:.2f}s)")

            engine = TranscriptionEngine()
            transcription = engine.transcribe_audio(
                audio_data,
                sr,
                max_simultaneous_notes=2
            )

            if not transcription:
                print("No notes detected.")
            else:
                for evt in transcription:
                    note_list = [n['note'] for n in evt['notes']]
                    print(
                        f"Onset: {evt['onset']:.3f}s | "
                        f"Offset: {evt['offset']:.3f}s | "
                        f"Duration: {evt['duration']:.3f}s | "
                        f"Notes: {note_list}"
                    )
        except Exception as e:
            print(f"Error while processing audio file: {e}")
