# TabCNN with Demucs Integration - OliTunes

This document explains how to use the TabCNN with Demucs integration for advanced guitar tablature generation in OliTunes.

## Overview

The TabCNN with Demucs integration combines two powerful machine learning models:

1. **Demucs**: A state-of-the-art music source separation model that can isolate guitar tracks from mixed audio
2. **TabCNN**: A convolutional neural network specifically designed for guitar tablature prediction

Together, these models enable OliTunes to generate highly accurate guitar tabs from full-mix audio files by:
- First separating the guitar track from other instruments
- Then analyzing the isolated guitar track to predict fret and string positions
- Finally rendering the predictions as human-readable tablature

## Features

- **Audio Source Separation**: Isolate guitar tracks from mixed audio using Demucs
- **Neural Tablature Generation**: Use TabCNN to predict tablature from audio
- **Method Comparison**: Compare traditional audio analysis with ML-based approaches
- **Model Training**: Train or fine-tune the TabCNN model on custom data

## Using the TabCNN Integration

### Processing Audio Files

1. Navigate to the **TabCNN + Demucs** tab in the OliTunes interface
2. Select an audio file from the dropdown menu
3. Click **Process with TabCNN** to start the analysis
4. Once processing is complete, view the generated tablature

### Comparing Tablature Methods

1. Select an audio file from the dropdown menu
2. Click the **Compare** button
3. View the side-by-side comparison between traditional and ML-based tablature
4. Check the comparative metrics to determine which method performed better

### Training the TabCNN Model

1. Navigate to the **TabCNN + Demucs** tab
2. Click the **Train TabCNN Model** button in the Model Training card
3. This will start a background process to:
   - Preprocess the GuitarSet data (if available)
   - Train the TabCNN model on the preprocessed data
   - Save the model weights for future use

## API Endpoints

The following API endpoints are available for programmatic access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze/cnn-tab/<filename>` | GET | Process an audio file with TabCNN and return tablature data |
| `/api/tabs/cnn/<filename>` | GET | Retrieve the text file representation of a generated tab |
| `/api/analyze/compare-tabs/<filename>` | GET | Compare traditional and CNN-based tablature methods |
| `/api/train/tabcnn` | POST | Start a background training process for the TabCNN model |

## Technical Details

### TabCNN Model

The TabCNN model is a convolutional neural network with the following characteristics:
- Input: Mel-spectrogram of audio (usually after Demucs separation)
- Output: A matrix of predictions (6×N) representing the 6 guitar strings across N time frames
- Training data: GuitarSet (a dataset of annotated guitar recordings)

### Demucs Model

Demucs is a music source separation model that:
- Separates audio into drums, bass, vocals, and other (which includes guitar)
- Uses a deep U-Net architecture with bidirectional LSTM
- Can isolate guitar parts even from complex mixes

### Processing Pipeline

1. Audio file is loaded and resampled to 22050Hz
2. Demucs separates the guitar track from other instruments
3. The separated track is converted to a mel-spectrogram
4. TabCNN processes the spectrogram to predict tablature
5. The raw predictions are converted to readable tablature format
6. The tablature is saved as a text file and returned

## Troubleshooting

### Common Issues

- **Missing Dependencies**: Ensure all required Python libraries are installed (tensorflow, demucs, etc.)
- **GPU Resources**: TabCNN processing can be slow without GPU acceleration
- **Low-quality Tablature**: For complex audio, try preprocessing with EQ to highlight guitar frequencies

### Improving Results

- Clean recordings produce better results than noisy ones
- Single-instrument recordings work better than full mixes
- Using Demucs significantly improves results on mixed audio
- Fine-tuning the model on your own guitar recordings can improve accuracy

## Future Improvements

- Integration with style-based positioning system
- Support for detecting advanced guitar techniques (bends, slides, etc.)
- User feedback collection to improve model performance
- A/B testing framework for comparing model versions
