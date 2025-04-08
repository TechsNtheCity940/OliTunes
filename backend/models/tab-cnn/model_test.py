"""
Enhanced TabCNN model testing script
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from TabCNN_GPU import TabCNN
from sklearn.metrics import accuracy_score, classification_report

# Configuration
CONFIG = {
    'data_path': r"f:\newrepos\olitunes\backend\models\tab-cnn\data",
    'spec_repr': 'cqt',
    'con_win_size': 9,
    'model_dir': r"f:\newrepos\olitunes\backend\models\tab-cnn\model\saved\c 2025-04-06 171455",
    'test_files': [
        r"c\00_Rock1-90-C#_solo.npz",
        r"c\00_Jazz2-110-Bb_comp.npz",
        r"c\00_Funk3-98-A_solo.npz",
        r"c\00_BN1-129-Eb_solo.npz",
        r"c\00_SS1-100-C#_comp.npz"
    ]
}

def load_model():
    """Load the trained model"""
    model = TabCNN(
        data_path=CONFIG['data_path'],
        spec_repr=CONFIG['spec_repr'],
        con_win_size=CONFIG['con_win_size']
    )
    checkpoint_path = os.path.join(CONFIG['model_dir'], "checkpoint.keras")
    model.model = tf.keras.models.load_model(checkpoint_path)
    return model

def ensure_evaluation_dir():
    """Ensure evaluation directory exists"""
    eval_dir = os.path.join(CONFIG['model_dir'], 'evaluation')
    try:
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir
    except Exception as e:
        print(f"Error creating evaluation directory: {str(e)}")
        return None

def visualize_predictions(predictions, true_tabs=None, filename=None):
    """Create and save guitar tab visualization"""
    try:
        strings = ['E', 'A', 'D', 'G', 'B', 'e']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Display predictions as a tab
        for i, (string, pred) in enumerate(zip(strings, predictions)):
            fret = np.argmax(pred)
            ax.text(0, i, f"{string} |--{fret if fret > 0 else '-'}--", 
                    fontfamily='monospace', fontsize=12)
        
        ax.axis('off')
        plt.title('Predicted Tablature')
        
        # Save to file if filename provided
        if filename:
            eval_dir = ensure_evaluation_dir()
            if eval_dir:
                output_path = os.path.join(eval_dir, f"pred_{filename}.png")
                try:
                    plt.savefig(output_path, bbox_inches='tight')
                    plt.close()
                    print(f"Visualization saved to: {os.path.abspath(output_path)}")
                    return True
                except Exception as e:
                    print(f"Error saving visualization: {str(e)}")
                    plt.close()
                    return False
            return False
        else:
            plt.show()
            return True
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        return False

def evaluate_model(model):
    """Comprehensive model evaluation"""
    all_preds = []
    all_true = []
    
    for file_rel_path in CONFIG['test_files']:
        file_path = os.path.join(CONFIG['data_path'], 'spec_repr', file_rel_path)
        try:
            data = np.load(file_path)
            input_data = data['repr']
            true_tab = data['tab']
            
            if len(input_data.shape) == 2:
                input_data = np.expand_dims(input_data, axis=0)
            
            preds = model.model.predict(input_data)
            
            # Store for metrics
            all_preds.append(preds[0])
            all_true.append(true_tab)
            
            # Visualize first 3 samples
            if len(all_preds) <= 3:
                print(f"\nSample: {os.path.basename(file_path)}")
                visualize_predictions(preds[0], true_tab, os.path.basename(file_path).split('.')[0])
            
        except Exception as e:
            print(f"Error with {file_path}: {str(e)}")
    
    # Calculate metrics
    if all_preds:
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        
        # Flatten for classification report
        y_true = all_true.reshape(-1)
        y_pred = all_preds.reshape(-1, 21).argmax(axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred):.2f}")
        
        # Per-string accuracy
        print("\nPer-string Accuracy:")
        for i in range(6):
            acc = accuracy_score(all_true[:,i], all_preds[:,i].argmax(axis=1))
            print(f"String {i+1}: {acc:.2f}")

if __name__ == "__main__":
    print("Loading model and running evaluation...")
    model = load_model()
    evaluate_model(model)
