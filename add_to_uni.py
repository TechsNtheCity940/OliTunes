def process_audio(self, audio_path: str, style: str = 'rock'):
    evaluator = ConfidenceEvaluator()
    # ... (after tabcnn prediction)
    use_pred, conf, rec = evaluator.should_use_prediction(raw_tab)
    if not use_pred and rec == 'use_rules':
        chords = self.audio_analyzer.detect_chords()  # Assume added to UnifiedTabProcessor
        refined_tab = self.text_gen.generate_from_chords(chords)
    else:
        refined_tab = self.lstm.predict(raw_tab)
        if rec == 'hybrid':
            rule_tab = self.text_gen.generate_from_chords(chords)
            refined_tab = evaluator.blend_predictions(refined_tab, rule_tab, conf)