# generate_midi.py
import mido
from mido import MidiFile, MidiTrack, Message
import time

def generate_test_midi(filename="ground_truth_test.mid"):
    """
    Generates a MIDI file containing a sequence of monophonic and polyphonic notes.
    
    Sequence:
    1. C4 (Monophonic, 1.0s)
    2. G4 (Monophonic, 1.0s)
    3. C4 + E4 + G4 (C Major Chord, Polyphonic, 1.0s)
    4. A3 + C4 (A Minor Dyad, Polyphonic, 0.5s)
    5. D4 (Monophonic, 0.5s)
    
    Parameters:
        filename (str): The name of the output MIDI file.
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo (required for mido to handle time correctly)
    # 500000 microseconds per beat (120 BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    track.append(Message('program_change', program=0, time=0)) # Piano instrument

    # Ticks per beat (TPB) is 480 by default in mido, which is used for timing calculation.
    TPB = mid.ticks_per_beat
    
    # Helper function to convert seconds to MIDI ticks
    def seconds_to_ticks(seconds, tempo_us=500000):
        # Time in ticks = (Time in seconds * Ticks per beat) / (Tempo in us / 1,000,000)
        # Assuming 500000 us/beat (120 BPM)
        return int(seconds * 120 / 60 * TPB) 

    # --- Test Sequence ---
    
    # 1. C4 (Monophonic) - Duration 1.0s
    track.append(Message('note_on', note=60, velocity=100, time=0))
    ticks = seconds_to_ticks(1.0)
    track.append(Message('note_off', note=60, velocity=0, time=ticks))

    # 2. G4 (Monophonic) - Duration 1.0s
    track.append(Message('note_on', note=67, velocity=100, time=0))
    ticks = seconds_to_ticks(1.0)
    track.append(Message('note_off', note=67, velocity=0, time=ticks))
    
    # 3. C Major Chord (Polyphonic) - C4, E4, G4 - Duration 1.0s
    chord_notes = [60, 64, 67] 
    track.append(Message('note_on', note=chord_notes[0], velocity=100, time=0)) # C4
    track.append(Message('note_on', note=chord_notes[1], velocity=100, time=0)) # E4
    track.append(Message('note_on', note=chord_notes[2], velocity=100, time=0)) # G4
    
    ticks = seconds_to_ticks(1.0)
    track.append(Message('note_off', note=chord_notes[0], velocity=0, time=ticks))
    track.append(Message('note_off', note=chord_notes[1], velocity=0, time=0)) # Same tick, time=0 relative to last event
    track.append(Message('note_off', note=chord_notes[2], velocity=0, time=0))
    
    # 4. A Minor Dyad (Polyphonic) - A3, C4 - Duration 0.5s
    dyad_notes = [57, 60] 
    track.append(Message('note_on', note=dyad_notes[0], velocity=100, time=0)) # A3
    track.append(Message('note_on', note=dyad_notes[1], velocity=100, time=0)) # C4
    
    ticks = seconds_to_ticks(0.5)
    track.append(Message('note_off', note=dyad_notes[0], velocity=0, time=ticks))
    track.append(Message('note_off', note=dyad_notes[1], velocity=0, time=0))
    
    # 5. D4 (Monophonic, Short) - Duration 0.5s
    track.append(Message('note_on', note=62, velocity=100, time=0))
    ticks = seconds_to_ticks(0.5)
    track.append(Message('note_off', note=62, velocity=0, time=ticks))

    # End of track
    track.append(mido.MetaMessage('end_of_track', time=1))
    
    # Save the file
    try:
        mid.save(filename)
        print(f"✅ Successfully created Ground Truth MIDI file: {filename}")
    except Exception as e:
        print(f"❌ Failed to save MIDI file: {e}")

if __name__ == '__main__':
    generate_test_midi()