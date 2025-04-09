class TabCNNProcessor:
    def __init__(self):
        self.model = self.build_model()
        self.model.load_weights('path/to/tabcnn_weights.h5')

    def build_model(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Add
        inputs = Input(shape=(128, 9, 1))  # Mel-spectrogram input
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        residual = x
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Add()([x, residual])  # Residual connection
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        note_output = Dense(6 * 21, activation='sigmoid')(x)  # Note presence
        fret_output = Dense(6 * 21, activation='softmax')(x)  # Fret positions
        model = Model(inputs, [note_output, fret_output])
        model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                      optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        return audio_data  # Already handled in UnifiedTabProcessor

    def predict_tablature(self, spec: np.ndarray) -> np.ndarray:
        note_preds, fret_preds = self.model.predict(spec[None, ...])  # Add batch dim
        note_preds = note_preds.reshape(-1, 6, 21) > 0.5
        fret_preds = fret_preds.reshape(-1, 6, 21)
        combined = np.zeros_like(fret_preds)
        for t in range(combined.shape[0]):
            for s in range(6):
                if note_preds[t, s, np.argmax(fret_preds[t, s])]:
                    combined[t, s] = fret_preds[t, s]
        return combined_