def process_audio(self, audio_path: str, style: str = 'rock'):
    # ... (after optimization)
    midi_converter = MidiConverter()
    midi_notes = [{'time': frame['time'], 'note': self.midi_to_note(fret, string), 'duration': 0.1}
                  for frame in self._create_fretboard_data(optimized_tab)
                  for note in frame['notes']]
    midi_path = midi_converter.notes_to_midi(midi_notes, 'output.mid', bpm=config.get('bpm', 120))
    results['guitar']['midi_path'] = midi_path