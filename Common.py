import os
import matplotlib.pyplot as plt

OUTPUT_LABELS = {
    'drama': 0, 'thriller': 1, 'adult': 2, 'documentary': 3, 'comedy': 4, 'crime': 5, 'reality-tv': 6, 
    'horror': 7, 'sport': 8, 'animation': 9, 'action': 10, 'fantasy': 11, 'short': 12, 'sci-fi': 13, 
    'music': 14, 'adventure': 15, 'talk-show': 16, 'western': 17, 'family': 18, 'mystery': 19, 'history': 20, 
    'news': 21, 'biography': 22, 'romance': 23, 'game-show': 24, 'musical': 25, 'war': 26
}

N_OUTPUT_LABELS = len(OUTPUT_LABELS)

# This function creates a directory if it deosn't already exists 
def wrap_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def plot_output_labels(output_labels):
    fig, ax = plt.subplots()
    ax.bar(output_labels.keys(), output_labels.values())
    ax.set_xticklabels(output_labels.keys(), rotation=90)
    ax.set_ylabel('Frequency')
    ax.set_title('Output Label Distribution')
    plt.show()