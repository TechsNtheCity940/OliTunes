def optimize_fretboard(self, predictions: np.ndarray, key: str = 'C'):
    analyzer = EnhancedAudioAnalyzer()
    notes = [{'note': self.midi_to_note(np.argmax(pred[string]), string), 'time': t * 0.0232}
             for t, pred in enumerate(predictions) for string in range(6)]
    mapped = analyzer.map_notes_to_fretboard(notes)
    optimized = np.zeros_like(predictions)
    for note in mapped:
        t = int(note['time'] / 0.0232)
        optimized[t, note['string']] = note['fret']
    return optimized