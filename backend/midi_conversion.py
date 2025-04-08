from audio_imports import *

class MidiConverter:
    """
    Handles conversion between audio analysis data and MIDI format.
    Creates playable MIDI files from detected notes.
    """
    
    def __init__(self):
        self.has_midi = PRETTY_MIDI_AVAILABLE and MIDO_AVAILABLE
        
    def notes_to_midi(self, notes: List[Dict[str, Any]], 
                      output_file: str, 
                      instrument_name: str = "Acoustic Guitar (nylon)",
                      bpm: float = 120.0) -> str:
        """
        Convert detected notes to a MIDI file
        
        Args:
            notes: List of note dictionaries with time, duration, note, and velocity
            output_file: Path to save the MIDI file
            instrument_name: MIDI instrument name
            bpm: Beats per minute for the MIDI file
        
        Returns:
            Path to the created MIDI file or error message
        """
        if not self.has_midi:
            return "MIDI libraries not available. Install pretty_midi and mido."
            
        if not notes:
            return "No notes provided for MIDI conversion"
            
        try:
            # Create a PrettyMIDI object
            midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            
            # Create an instrument
            instrument = pretty_midi.Instrument(
                program=pretty_midi.instrument_name_to_program(instrument_name)
            )
            
            # Add notes to the instrument
            for note_data in notes:
                # Skip if missing required fields
                if not all(k in note_data for k in ['note', 'time', 'duration']):
                    continue
                    
                # Extract note properties
                note_name = note_data['note']
                start_time = note_data['time']
                duration = note_data['duration']
                velocity = int(note_data.get('velocity', 64))  # Default velocity if not provided
                
                # Convert note name to MIDI note number
                try:
                    if isinstance(note_name, int):
                        note_number = note_name
                    else:
                        note_number = pretty_midi.note_name_to_number(note_name)
                except:
                    # Try librosa's conversion if pretty_midi fails
                    try:
                        note_number = int(round(librosa.note_to_midi(note_name)))
                    except:
                        logger.warning(f"Could not convert note name: {note_name}")
                        continue
                
                # Create the note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_number,
                    start=start_time,
                    end=start_time + duration
                )
                
                # Add to instrument
                instrument.notes.append(note)
            
            # Add instrument to the MIDI file
            midi.instruments.append(instrument)
            
            # Write out the MIDI file
            midi.write(output_file)
            
            return output_file
            
        except Exception as e:
            error_msg = f"Error creating MIDI file: {str(e)}"
            logger.error(error_msg)
            return error_msg
            
    def midi_to_notes(self, midi_file: str) -> List[Dict[str, Any]]:
        """
        Extract notes from a MIDI file
        
        Args:
            midi_file: Path to the MIDI file
            
        Returns:
            List of note dictionaries
        """
        if not self.has_midi:
            logger.error("MIDI libraries not available. Install pretty_midi and mido.")
            return []
            
        try:
            # Load the MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            
            notes = []
            
            # Process each instrument
            for i, instrument in enumerate(midi_data.instruments):
                instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
                
                # Process each note
                for note in instrument.notes:
                    note_name = pretty_midi.note_number_to_name(note.pitch)
                    
                    notes.append({
                        'note': note_name,
                        'time': float(note.start),
                        'duration': float(note.end - note.start),
                        'velocity': int(note.velocity),
                        'frequency': float(librosa.midi_to_hz(note.pitch)),
                        'instrument': instrument_name,
                        'instrument_idx': i
                    })
            
            # Sort by start time
            notes.sort(key=lambda x: x['time'])
            
            return notes
            
        except Exception as e:
            logger.error(f"Error reading MIDI file: {str(e)}")
            return []
            
    def create_midi_drum_pattern(self, 
                               pattern: List[Dict[str, Any]], 
                               output_file: str,
                               bpm: float = 120.0) -> str:
        """
        Create a MIDI drum pattern
        
        Args:
            pattern: List of drum hit dictionaries with time, drum_type, and velocity
            output_file: Path to save the MIDI file
            bpm: Beats per minute
            
        Returns:
            Path to the created MIDI file or error message
        """
        if not self.has_midi:
            return "MIDI libraries not available. Install pretty_midi and mido."
            
        if not pattern:
            return "No drum pattern provided"
            
        try:
            # Create a PrettyMIDI object
            midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            
            # Create a drum instrument (channel 10)
            drums = pretty_midi.Instrument(program=0, is_drum=True)
            
            # MIDI drum map
            drum_map = {
                'kick': 36,       # Bass Drum 1
                'snare': 38,      # Acoustic Snare
                'hi-hat': 42,     # Closed Hi-Hat
                'open-hat': 46,   # Open Hi-Hat
                'tom1': 48,       # Hi Mid Tom
                'tom2': 45,       # Low Tom
                'crash': 49,      # Crash Cymbal 1
                'ride': 51,       # Ride Cymbal 1
                'clap': 39,       # Hand Clap
                'rim': 37,        # Side Stick
            }
            
            # Add drum hits to the instrument
            for hit in pattern:
                # Skip if missing required fields
                if not all(k in hit for k in ['time', 'drum_type']):
                    continue
                    
                # Extract hit properties
                drum_type = hit['drum_type']
                start_time = hit['time']
                velocity = int(hit.get('velocity', 100))  # Default velocity
                duration = hit.get('duration', 0.1)  # Default short duration
                
                # Get the MIDI note number for this drum
                if drum_type in drum_map:
                    note_number = drum_map[drum_type]
                elif isinstance(drum_type, int) and 0 <= drum_type <= 127:
                    # Allow direct MIDI note numbers
                    note_number = drum_type
                else:
                    logger.warning(f"Unknown drum type: {drum_type}")
                    continue
                
                # Create the note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_number,
                    start=start_time,
                    end=start_time + duration
                )
                
                # Add to instrument
                drums.notes.append(note)
            
            # Add drums to the MIDI file
            midi.instruments.append(drums)
            
            # Write out the MIDI file
            midi.write(output_file)
            
            return output_file
            
        except Exception as e:
            error_msg = f"Error creating drum MIDI file: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def add_chord_track(self, 
                       midi_file: str, 
                       chords: List[Dict[str, Any]],
                       instrument_name: str = "Acoustic Guitar (steel)",
                       velocity: int = 80) -> str:
        """
        Add a chord track to an existing MIDI file
        
        Args:
            midi_file: Path to the MIDI file to modify
            chords: List of chord dictionaries with time, duration, and notes
            instrument_name: MIDI instrument name for the chord track
            velocity: Velocity for chord notes
            
        Returns:
            Path to the modified MIDI file or error message
        """
        if not self.has_midi:
            return "MIDI libraries not available. Install pretty_midi and mido."
            
        if not chords:
            return "No chords provided"
            
        try:
            # Load the MIDI file
            midi = pretty_midi.PrettyMIDI(midi_file)
            
            # Create a new instrument for chords
            chord_instrument = pretty_midi.Instrument(
                program=pretty_midi.instrument_name_to_program(instrument_name)
            )
            
            # Add chords to the instrument
            for chord in chords:
                # Skip if missing required fields
                if not all(k in chord for k in ['time', 'duration', 'notes']):
                    continue
                    
                # Extract chord properties
                start_time = chord['time']
                duration = chord['duration']
                chord_notes = chord['notes']
                
                # Add each note in the chord
                for note_name in chord_notes:
                    try:
                        if isinstance(note_name, int):
                            note_number = note_name
                        else:
                            note_number = pretty_midi.note_name_to_number(note_name)
                    except:
                        # Try librosa's conversion if pretty_midi fails
                        try:
                            note_number = int(round(librosa.note_to_midi(note_name)))
                        except:
                            logger.warning(f"Could not convert note name: {note_name}")
                            continue
                    
                    # Create the note
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=note_number,
                        start=start_time,
                        end=start_time + duration
                    )
                    
                    # Add to chord instrument
                    chord_instrument.notes.append(note)
            
            # Add chord instrument to the MIDI file
            midi.instruments.append(chord_instrument)
            
            # Write out the modified MIDI file
            midi.write(midi_file)
            
            return midi_file
            
        except Exception as e:
            error_msg = f"Error adding chord track: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def create_midi_from_notes(self, notes: List[Dict[str, Any]], 
                         output_path: str, 
                         tempo: float = 120.0,
                         instrument_name: str = "Acoustic Guitar (nylon)") -> Dict[str, Any]:
        """
        Create a MIDI file from detected notes.
        
        Args:
            notes: List of note dictionaries with time, duration, and midi or note info
            output_path: Path to save the MIDI file
            tempo: Tempo in BPM
            instrument_name: MIDI instrument name
        
        Returns:
            Dictionary with success status and file path or error message
        """
        try:
            # Check if required libraries are available
            if not PRETTY_MIDI_AVAILABLE:
                logger.warning("pretty_midi not available for MIDI creation")
                return {
                    'success': False,
                    'error': 'MIDI libraries not available. Install pretty_midi.'
                }
                
            # Ensure we have valid notes
            if not notes:
                return {
                    'success': False,
                    'error': 'No notes provided for MIDI conversion'
                }
            
            # Create a PrettyMIDI object
            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
            
            # Create an instrument
            try:
                program_num = pretty_midi.instrument_name_to_program(instrument_name)
            except:
                logger.warning(f"Unknown instrument name: {instrument_name}. Using acoustic guitar.")
                program_num = 24  # Acoustic Guitar (nylon)
                
            instrument = pretty_midi.Instrument(program=program_num)
            
            # Add notes to the instrument
            for note_data in notes:
                # Skip if missing required fields
                if not ('midi' in note_data or 'note' in note_data) or 'time' not in note_data:
                    continue
                    
                # Get note pitch in MIDI format
                if 'midi' in note_data:
                    pitch = int(note_data['midi'])
                else:
                    try:
                        # Convert note name (e.g., 'C4') to MIDI
                        pitch = librosa.note_to_midi(note_data['note'])
                    except:
                        logger.warning(f"Could not convert note {note_data['note']} to MIDI pitch. Skipping.")
                        continue
            
                # Get start time
                start_time = float(note_data['time'])
                
                # Get duration (default to 0.25 seconds if not provided)
                if 'duration' in note_data:
                    duration = float(note_data['duration'])
                else:
                    duration = 0.25
                    
                # Get velocity (default to 100 if not provided)
                if 'velocity' in note_data:
                    velocity = int(note_data['velocity'])
                else:
                    velocity = 100
                    
                # Create a Note object
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=start_time + duration
                )
                
                # Add it to the instrument
                instrument.notes.append(note)
            
            # Add the instrument to the PrettyMIDI object
            midi.instruments.append(instrument)
            
            # Write the MIDI file
            midi.write(output_path)
            
            return {
                'success': True,
                'path': output_path,
                'note_count': len(instrument.notes)
            }
            
        except Exception as e:
            logger.error(f"Error in MIDI creation: {str(e)}")
            return {
                'success': False,
                'error': f"MIDI creation failed: {str(e)}"
            }

# Standalone function for direct import
def create_midi_from_notes(notes: List[Dict[str, Any]], 
                         output_path: str, 
                         tempo: float = 120.0,
                         instrument_name: str = "Acoustic Guitar (nylon)") -> Dict[str, Any]:
    """
    Create a MIDI file from detected notes.
    
    Args:
        notes: List of note dictionaries with time, duration, and midi or note info
        output_path: Path to save the MIDI file
        tempo: Tempo in BPM
        instrument_name: MIDI instrument name
        
    Returns:
        Dictionary with success status and file path or error message
    """
    # Create converter instance and call its method
    converter = MidiConverter()
    return converter.create_midi_from_notes(
        notes=notes, 
        output_path=output_path, 
        tempo=tempo, 
        instrument_name=instrument_name
    )
