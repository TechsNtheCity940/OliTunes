# In app.py
analyzer = MusicTheoryAnalyzer()
analysis = {
    'key': analyzer.detect_key(y, sr)['key'],
    'tempo': librosa.beat.tempo(y, sr=sr)[0],
    'chords': analyzer.analyze_chord_progression(y, sr)
}
tabs = processor.process_audio(temp_path, style, key=analysis['key'])