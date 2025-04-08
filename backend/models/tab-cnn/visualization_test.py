"""
Standalone visualization test
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def show_tab(predictions):
    """Display guitar tab visualization"""
    strings = ['E', 'A', 'D', 'G', 'B', 'e']
    
    # Create figure with adjusted size
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Add space at top and bottom
    fig.subplots_adjust(top=0.85, bottom=0.15)
    
    # Sample predictions (6 strings x 21 frets)
    for i, (string, pred) in enumerate(zip(strings, predictions)):
        fret = np.argmax(pred)
        ax.text(0, i, f"{string} |--{fret if fret > 0 else '-'}--", 
                fontfamily='monospace', fontsize=12)
    
    ax.axis('off')
    plt.title('Test Tablature Visualization', pad=20)
    
    # Ensure window appears
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()

if __name__ == "__main__":
    # Create random predictions (6 strings x 21 frets)
    test_preds = np.random.rand(6, 21)
    show_tab(test_preds)
