from audio_imports import *

class TablatureGenerator:
    """
    Generates guitar and bass tablature from detected notes.
    Optimizes fretboard positions for playability.
    """
    
    # Define constants for guitar string tunings
    STANDARD_GUITAR_TUNING = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']  # Low to high
    STANDARD_BASS_TUNING = ['E1', 'A1', 'D2', 'G2']  # Low to high
    DROP_D_TUNING = ['D2', 'A2', 'D3', 'G3', 'B3', 'E4']
    
    # Define guitar fretboard range
    MAX_FRET = 22
    
    def __init__(self, instrument: str = 'guitar', tuning: List[str] = None):
        """
        Initialize the tablature generator
        
        Args:
            instrument: 'guitar' or 'bass'
            tuning: List of string tunings from low to high (defaults to standard)
        """
        self.instrument = instrument.lower()
        
        # Set default tuning based on instrument
        if tuning is None:
            if self.instrument == 'guitar':
                self.tuning = self.STANDARD_GUITAR_TUNING
            elif self.instrument == 'bass':
                self.tuning = self.STANDARD_BASS_TUNING
            else:
                raise ValueError(f"Unsupported instrument: {instrument}")
        else:
            self.tuning = tuning
        
        # Convert tuning notes to MIDI note numbers for easier calculation
        self.tuning_midi = []
        for note in self.tuning:
            try:
                self.tuning_midi.append(librosa.note_to_midi(note))
            except:
                logger.warning(f"Could not convert note {note} to MIDI number")
                # Default to a reasonable value
                self.tuning_midi.append(40)  # Default to E1
        
        # Create a mapping of MIDI notes to string+fret combinations
        self.note_to_fret_map = self._create_fretboard_map()
    
    def _create_fretboard_map(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Create a mapping of MIDI note numbers to possible string+fret combinations
        
        Returns:
            Dictionary of MIDI notes to list of (string_index, fret) tuples
        """
        fretboard_map = defaultdict(list)
        
        # For each string
        for string_idx, open_note in enumerate(self.tuning_midi):
            # For each fret position
            for fret in range(self.MAX_FRET + 1):
                note = open_note + fret
                fretboard_map[note].append((string_idx, fret))
        
        return fretboard_map
    
    def generate_tab(self, notes: List[Dict[str, Any]], optimize: bool = True) -> Dict[str, Any]:
        """
        Generate tablature from a list of notes
        
        Args:
            notes: List of note dictionaries with time, note, and duration
            optimize: Whether to optimize for playability
            
        Returns:
            Dictionary with tablature information
        """
        if not notes:
            return {'error': 'No notes provided'}
        
        # Sort notes by time
        sorted_notes = sorted(notes, key=lambda x: x.get('time', 0))
        
        # Convert note names to MIDI note numbers
        midi_notes = []
        for note_data in sorted_notes:
            note_name = note_data.get('note')
            if note_name:
                try:
                    if isinstance(note_name, int):
                        midi_note = note_name
                    else:
                        midi_note = librosa.note_to_midi(note_name)
                    
                    midi_notes.append({
                        'midi': midi_note,
                        'time': note_data.get('time', 0),
                        'duration': note_data.get('duration', 0.25),
                        'velocity': note_data.get('velocity', 64)
                    })
                except:
                    logger.warning(f"Could not convert note {note_name} to MIDI number")
        
        if not midi_notes:
            return {'error': 'Could not convert any notes to MIDI numbers'}
        
        # If optimize is True, find optimal fret positions
        if optimize:
            tab_positions = self._optimize_fret_positions(midi_notes)
        else:
            # Otherwise, just use the first available position for each note
            tab_positions = []
            for note in midi_notes:
                midi = note['midi']
                if midi in self.note_to_fret_map:
                    string_idx, fret = self.note_to_fret_map[midi][0]
                    tab_positions.append({
                        'midi': midi,
                        'string': string_idx,
                        'fret': fret,
                        'time': note['time'],
                        'duration': note['duration'],
                        'velocity': note['velocity']
                    })
        
        # Organize into measures based on time
        measures = self._organize_into_measures(tab_positions)
        
        # Convert to ASCII tab format
        ascii_tab = self._create_ascii_tab(measures)
        
        return {
            'positions': tab_positions,
            'measures': measures,
            'ascii_tab': ascii_tab,
            'tuning': self.tuning
        }
    
    def _optimize_fret_positions(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize fret positions for playability
        
        Args:
            notes: List of note dictionaries with midi, time, and duration
            
        Returns:
            List of note dictionaries with string and fret positions
        """
        # If no notes, return empty list
        if not notes:
            return []
        
        # If only one note, use the lowest possible fret position
        if len(notes) == 1:
            midi = notes[0]['midi']
            if midi in self.note_to_fret_map:
                # Sort by fret number to prefer lower frets
                positions = sorted(self.note_to_fret_map[midi], key=lambda x: x[1])
                string_idx, fret = positions[0]
                return [{
                    'midi': midi,
                    'string': string_idx,
                    'fret': fret,
                    'time': notes[0]['time'],
                    'duration': notes[0]['duration'],
                    'velocity': notes[0]['velocity']
                }]
            else:
                return []
        
        # Group notes by time to handle chords
        notes_by_time = defaultdict(list)
        for note in notes:
            notes_by_time[note['time']].append(note)
        
        # Sort times
        times = sorted(notes_by_time.keys())
        
        # Find optimal positions for each time slice
        optimized_positions = []
        last_position = None
        
        for time in times:
            time_notes = notes_by_time[time]
            
            # Is this a chord (multiple notes at the same time)?
            is_chord = len(time_notes) > 1
            
            # For each note at this time
            chord_positions = []
            for note in time_notes:
                midi = note['midi']
                
                if midi not in self.note_to_fret_map:
                    # No valid position for this note
                    continue
                
                # Get all possible positions
                positions = self.note_to_fret_map[midi]
                
                if is_chord:
                    # For chords, optimize for lowest average fret position
                    # and make sure notes can be played simultaneously
                    chord_positions.append({
                        'midi': midi,
                        'positions': positions,
                        'time': note['time'],
                        'duration': note['duration'],
                        'velocity': note['velocity']
                    })
                else:
                    # For single notes, optimize for minimizing hand movement
                    if last_position is None:
                        # First note, use lowest fret
                        positions = sorted(positions, key=lambda x: x[1])
                        string_idx, fret = positions[0]
                    else:
                        # Find closest position to last one
                        last_string, last_fret = last_position
                        
                        # Calculate distance to each potential position
                        distances = []
                        for string_idx, fret in positions:
                            # Distance is a combination of string and fret distance
                            string_distance = abs(string_idx - last_string)
                            fret_distance = abs(fret - last_fret)
                            
                            # Weight fret distance more heavily
                            distance = fret_distance * 1.5 + string_distance
                            distances.append((distance, string_idx, fret))
                        
                        # Sort by distance
                        distances.sort()
                        
                        # Use the closest position
                        _, string_idx, fret = distances[0]
                    
                    # Add to optimized positions
                    optimized_positions.append({
                        'midi': midi,
                        'string': string_idx,
                        'fret': fret,
                        'time': note['time'],
                        'duration': note['duration'],
                        'velocity': note['velocity']
                    })
                    
                    # Update last position
                    last_position = (string_idx, fret)
            
            # If we have a chord, find compatible positions
            if chord_positions:
                chord_solution = self._optimize_chord_positions(chord_positions)
                optimized_positions.extend(chord_solution)
                
                # Update last position based on the highest string used in the chord
                if chord_solution:
                    highest_string = min([note['string'] for note in chord_solution])
                    last_fret = next((note['fret'] for note in chord_solution if note['string'] == highest_string), 0)
                    last_position = (highest_string, last_fret)
        
        return optimized_positions
    
    def _optimize_chord_positions(self, chord_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find compatible positions for a chord
        
        Args:
            chord_notes: List of notes in the chord with possible positions
            
        Returns:
            List of notes with optimized string and fret positions
        """
        # Special case for one note "chord"
        if len(chord_notes) == 1:
            note = chord_notes[0]
            positions = sorted(note['positions'], key=lambda x: x[1])  # Sort by fret
            string_idx, fret = positions[0]  # Use lowest fret
            
            return [{
                'midi': note['midi'],
                'string': string_idx,
                'fret': fret,
                'time': note['time'],
                'duration': note['duration'],
                'velocity': note['velocity']
            }]
        
        # Try different combinations of positions
        best_score = float('inf')
        best_solution = None
        
        # Get all possible combinations
        position_options = [note['positions'] for note in chord_notes]
        
        # Limit the number of combinations to try
        max_combinations = 100
        total_combinations = np.prod([len(pos) for pos in position_options])
        
        if total_combinations > max_combinations:
            # Too many combinations, limit each note to 2 positions
            position_options = [sorted(pos, key=lambda x: x[1])[:2] for pos in position_options]
        
        # Generate all combinations
        for positions in itertools.product(*position_options):
            # Check if this combination is playable
            strings_used = set()
            max_fret = 0
            min_fret = float('inf')
            
            is_valid = True
            for (string_idx, fret) in positions:
                # Check if string is already used
                if string_idx in strings_used:
                    is_valid = False
                    break
                
                strings_used.add(string_idx)
                max_fret = max(max_fret, fret)
                min_fret = min(min_fret, fret)
            
            # Check fret span (is it playable by human hand?)
            fret_span = max_fret - min_fret
            if fret_span > 4 and min_fret > 0:  # Allow open strings with wide spans
                is_valid = False
            
            if is_valid:
                # Calculate score based on fret positions and spread
                # Lower is better
                score = max_fret + fret_span
                
                if score < best_score:
                    best_score = score
                    best_solution = positions
        
        # If no valid solution, fall back to individual note optimization
        if best_solution is None:
            return []
        
        # Convert solution to the standard format
        solution = []
        for i, (string_idx, fret) in enumerate(best_solution):
            note = chord_notes[i]
            solution.append({
                'midi': note['midi'],
                'string': string_idx,
                'fret': fret,
                'time': note['time'],
                'duration': note['duration'],
                'velocity': note['velocity']
            })
        
        return solution
    
    def _organize_into_measures(self, tab_positions: List[Dict[str, Any]], 
                              beats_per_measure: int = 4) -> List[List[Dict[str, Any]]]:
        """
        Organize tab positions into measures
        
        Args:
            tab_positions: List of note dictionaries with string and fret positions
            beats_per_measure: Number of beats per measure
            
        Returns:
            List of measures, each containing a list of notes
        """
        if not tab_positions:
            return []
        
        # Sort by time
        sorted_positions = sorted(tab_positions, key=lambda x: x['time'])
        
        # Find the first and last times
        first_time = sorted_positions[0]['time']
        last_time = sorted_positions[-1]['time']
        
        # Estimate beat duration based on note durations
        durations = [note['duration'] for note in sorted_positions if note['duration'] > 0]
        if durations:
            beat_duration = min(durations) * 4  # Assume shortest note is a 16th note
        else:
            beat_duration = 0.5  # Default
        
        # Calculate measure duration
        measure_duration = beat_duration * beats_per_measure
        
        # Calculate number of measures
        total_duration = last_time - first_time
        num_measures = max(1, int(np.ceil(total_duration / measure_duration)))
        
        # Initialize measures
        measures = [[] for _ in range(num_measures)]
        
        # Assign notes to measures
        for note in sorted_positions:
            measure_idx = int((note['time'] - first_time) / measure_duration)
            
            # Handle potential rounding errors
            if measure_idx >= num_measures:
                measure_idx = num_measures - 1
            
            # Add note to measure
            measures[measure_idx].append(note)
        
        return measures
    
    def _create_ascii_tab(self, measures: List[List[Dict[str, Any]]]) -> str:
        """
        Create ASCII tablature representation
        
        Args:
            measures: List of measures, each containing a list of notes
            
        Returns:
            ASCII tablature as a string
        """
        if not measures:
            return "No notes to display"
        
        # Create header with tuning
        header = "Tuning: " + " ".join(self.tuning) + "\n\n"
        
        # Initialize tab with empty strings
        tab_lines = [[] for _ in range(len(self.tuning))]
        
        # Process each measure
        for measure_idx, measure in enumerate(measures):
            # Sort notes in measure by time
            sorted_notes = sorted(measure, key=lambda x: x['time'])
            
            # Create a mapping of (string, position) -> fret number
            # This handles multiple notes at the same position
            string_positions = defaultdict(list)
            max_position = 0
            
            # First pass: calculate positions
            last_time = None
            position = 0
            
            for note in sorted_notes:
                time = note['time']
                
                # If time changed, update position
                if last_time is not None and time > last_time:
                    position += 2  # Add space between different times
                
                # Update position mapping
                string_positions[(note['string'], position)].append(note['fret'])
                
                # Update tracking variables
                last_time = time
                max_position = max(max_position, position)
            
            # Create measure separator
            for i in range(len(self.tuning)):
                tab_lines[i].append("|")
            
            # Second pass: fill in the tab
            for pos in range(max_position + 1):
                # For each string
                for string_idx in range(len(self.tuning)):
                    # Check if there's a note at this position
                    if (string_idx, pos) in string_positions:
                        frets = string_positions[(string_idx, pos)]
                        
                        # Format fret number (handle double-digit frets)
                        if len(frets) == 1:
                            fret_str = str(frets[0]).rjust(2)
                        else:
                            # Multiple notes at same position (shouldn't happen often)
                            fret_str = "XX"
                    else:
                        # No note, add a spacer
                        fret_str = "--"
                    
                    tab_lines[string_idx].append(fret_str)
            
            # Add measure end
            for i in range(len(self.tuning)):
                tab_lines[i].append("|")
        
        # Combine all lines
        tab_text = header
        
        # Add string labels
        for i, tuning_note in enumerate(self.tuning):
            line = tuning_note[0] + "|" + "".join(tab_lines[i]) + "\n"
            tab_text += line
        
        return tab_text
    
    def optimize_fingering(self, tab_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign left-hand fingerings to tab positions
        
        Args:
            tab_positions: List of note dictionaries with string and fret positions
            
        Returns:
            Same list with finger assignments added
        """
        if not tab_positions:
            return []
        
        # Sort by time
        sorted_positions = sorted(tab_positions, key=lambda x: x['time'])
        
        # Find notes that are played at the same time (chords)
        notes_by_time = defaultdict(list)
        for note in sorted_positions:
            notes_by_time[note['time']].append(note)
        
        # Process each time slice
        for time, notes in notes_by_time.items():
            # Is this a chord?
            if len(notes) > 1:
                # Sort by string (low to high)
                notes.sort(key=lambda x: x['string'], reverse=True)
                
                # Find the fret span
                min_fret = min([note['fret'] for note in notes if note['fret'] > 0])
                max_fret = max([note['fret'] for note in notes])
                
                # Skip open strings when assigning fingers
                fretted_notes = [note for note in notes if note['fret'] > 0]
                
                # Assign fingers based on fret span
                if fretted_notes:
                    fret_span = max_fret - min_fret
                    
                    if fret_span <= 3:
                        # Can use one finger per fret
                        for note in fretted_notes:
                            # Assign finger (1=index, 2=middle, 3=ring, 4=pinky)
                            finger = note['fret'] - min_fret + 1
                            note['finger'] = min(finger, 4)  # Limit to 4 fingers
                    else:
                        # Needs stretching or position shift
                        # Simple algorithm: assign lowest frets to index finger,
                        # highest to pinky, and distribute middle and ring
                        fretted_notes.sort(key=lambda x: x['fret'])
                        
                        if len(fretted_notes) <= 4:
                            # One finger per note
                            for i, note in enumerate(fretted_notes):
                                note['finger'] = i + 1
                        else:
                            # More notes than fingers, need to use barre or multiple notes per finger
                            for i, note in enumerate(fretted_notes):
                                if i < 3:
                                    note['finger'] = i + 1
                                else:
                                    note['finger'] = 4  # Assign rest to pinky
                
                # Mark open strings
                for note in notes:
                    if note['fret'] == 0:
                        note['finger'] = 0  # 0 = open string
            
            else:
                # Single note
                note = notes[0]
                if note['fret'] == 0:
                    note['finger'] = 0  # Open string
                else:
                    # For single notes, default to first finger for low frets,
                    # middle finger for mid frets, etc.
                    if note['fret'] <= 2:
                        note['finger'] = 1
                    elif note['fret'] <= 4:
                        note['finger'] = 2
                    elif note['fret'] <= 6:
                        note['finger'] = 3
                    else:
                        note['finger'] = 4
        
        return sorted_positions
    
    def tab_to_svg(self, tab_data: Dict[str, Any], width: int = 800, height: int = None) -> str:
        """
        Convert tablature data to SVG format for display
        
        Args:
            tab_data: Dictionary with tab information from generate_tab
            width: SVG width in pixels
            height: SVG height in pixels (calculated if None)
            
        Returns:
            SVG string representation of the tablature
        """
        if 'measures' not in tab_data or not tab_data['measures']:
            return "<svg>No tablature data</svg>"
        
        measures = tab_data['measures']
        tuning = tab_data.get('tuning', self.tuning)
        
        # Calculate dimensions
        string_count = len(tuning)
        measure_count = len(measures)
        
        # Layout parameters
        string_spacing = 12
        measure_width = 180
        margin = 40
        
        # Calculate measures per row based on width
        measures_per_row = max(1, min(4, int((width - 2 * margin) / measure_width)))
        
        # Calculate rows needed
        rows = math.ceil(measure_count / measures_per_row)
        
        # Calculate height if not specified
        if height is None:
            height = margin * 2 + rows * (string_count * string_spacing + 30)
        
        # Start SVG
        svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add style
        svg += '<style>\n'
        svg += '.tab-string { stroke: black; stroke-width: 1; }\n'
        svg += '.tab-measure { stroke: black; stroke-width: 1; }\n'
        svg += '.tab-text { font-family: monospace; font-size: 12px; }\n'
        svg += '.tab-tuning { font-family: monospace; font-size: 12px; text-anchor: end; }\n'
        svg += '.tab-fret { font-family: monospace; font-size: 12px; text-anchor: middle; }\n'
        svg += '.tab-finger { font-family: monospace; font-size: 8px; fill: blue; text-anchor: middle; }\n'
        svg += '</style>\n'
        
        # Draw each row
        for row in range(rows):
            row_y = margin + row * (string_count * string_spacing + 30)
            
            # Draw tuning labels
            for i, note in enumerate(tuning):
                string_y = row_y + i * string_spacing
                svg += f'<text x="{margin - 10}" y="{string_y + 4}" class="tab-tuning">{note}</text>\n'
            
            # Draw measures for this row
            start_measure = row * measures_per_row
            end_measure = min(measure_count, start_measure + measures_per_row)
            
            for m, measure_idx in enumerate(range(start_measure, end_measure)):
                measure_x = margin + m * measure_width
                
                # Draw measure lines
                measure_height = (string_count - 1) * string_spacing
                svg += f'<line x1="{measure_x}" y1="{row_y}" x2="{measure_x}" y2="{row_y + measure_height}" class="tab-measure" />\n'
                svg += f'<line x1="{measure_x + measure_width}" y1="{row_y}" x2="{measure_x + measure_width}" y2="{row_y + measure_height}" class="tab-measure" />\n'
                
                # Draw strings
                for i in range(string_count):
                    string_y = row_y + i * string_spacing
                    svg += f'<line x1="{measure_x}" y1="{string_y}" x2="{measure_x + measure_width}" y2="{string_y}" class="tab-string" />\n'
                
                # Draw notes in this measure
                if measure_idx < len(measures):
                    notes = measures[measure_idx]
                    
                    # Group notes by time
                    notes_by_time = defaultdict(list)
                    for note in notes:
                        notes_by_time[note['time']].append(note)
                    
                    # Sort times
                    times = sorted(notes_by_time.keys())
                    
                    # Map times to x-positions
                    if times:
                        time_range = max(times) - min(times)
                        
                        # Scale the width based on the longest measure
                        if time_range > 0:
                            scale = (measure_width - 20) / time_range
                        else:
                            scale = 1
                        
                        # Draw each note
                        for time in times:
                            time_notes = notes_by_time[time]
                            x_pos = measure_x + 10 + (time - min(times)) * scale
                            
                            for note in time_notes:
                                string_idx = note['string']
                                string_y = row_y + string_idx * string_spacing
                                
                                # Draw fret number
                                svg += f'<text x="{x_pos}" y="{string_y + 4}" class="tab-fret">{note["fret"]}</text>\n'
                                
                                # Draw fingering if available
                                if 'finger' in note and note['finger'] > 0:
                                    svg += f'<text x="{x_pos}" y="{string_y - 5}" class="tab-finger">{note["finger"]}</text>\n'
                
                # Add measure number
                svg += f'<text x="{measure_x + measure_width/2}" y="{row_y - 10}" class="tab-text" text-anchor="middle">{measure_idx + 1}</text>\n'
        
        # Close SVG
        svg += '</svg>'
        
        return svg
