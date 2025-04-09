Fretboard Position Model
=======================

This model predicts optimal guitar fretboard positions based on musical context.
It's used by the OliTunes tablature generation system.

Files:
- model_weights.h5: Neural network weights (when trained)
- model_architecture.json: Network architecture definition
- metadata.json: Model metadata
- model_ready: Marker file indicating the model is ready for use

Input Features:
- MIDI note (one-hot encoding)
- Previous fret position
- String number
- Key context
- Playing style

Output:
- Probability distribution over possible string/fret combinations

The model helps optimize tablature generation by considering:
1. Musical context (key, style)
2. Previous hand positions for minimal movement
3. Playability and ergonomic considerations
4. Common chord shapes and patterns
