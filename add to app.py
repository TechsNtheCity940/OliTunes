@app.route('/generate_tab', methods=['POST'])
def generate_tab():
    """Full pipeline: audio -> separation -> tab prediction."""
    try:
        audio_file = request.files['audio']
        style = request.form.get('style', 'rock')
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
        audio_file.save(temp_path)

        # Analyze audio for context
        analyzer = AudioAnalyzer(temp_path)
        analysis = {
            'key': analyzer.detect_key().get('key', 'C'),
            'tempo': analyzer.detect_tempo(),
            'style': style
        }

        # Process with UnifiedTabProcessor
        processor = UnifiedTabProcessor()
        tabs = processor.process_audio(temp_path, style)

        if not tabs:
            # Fallback: pattern-based tabs
            chords = analyzer.detect_chords()
            tab_data = {'bars': [], 'notePositions': []}  # Simplified fallback
            text_gen = TabTextGenerator()
            tabs = {'guitar': {'text_tab': text_gen.generate_from_chords(chords)}}

        # Format output
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'tabs': tabs,
            'visualization': processor._create_fretboard_data(tabs['guitar'].get('predictions', np.array([])))
        })
    except Exception as e:
        logger.error(f"Tab generation failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500