from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(true_labels, predicted_labels):
    """Evaluate model performance."""
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return precision, recall, f1

if __name__ == "__main__":
    # Placeholder: Load true and predicted labels
    true_labels = np.random.randint(0, 5, 1000)  # Replace with actual labels
    predicted_labels = np.random.randint(0, 5, 1000)  # Replace with actual predictions
    precision, recall, f1 = evaluate_model(true_labels, predicted_labels)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F-measure: {f1:.2f}")