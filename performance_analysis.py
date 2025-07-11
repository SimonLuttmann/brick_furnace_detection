from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_and_log_performance(all_labels, all_preds, num_classes, epoch, writer=None):
    """
    Calculate and log performance metrics from predictions and true labels.
    
    Args:
        all_labels: numpy array of true labels
        all_preds: numpy array of predicted labels
        num_classes: number of classes
        epoch: current epoch number
        writer: tensorboard writer (optional)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate per-class metrics from confusion matrix
    print("\nPer-class metrics:")
        
    # Optional: Get precision, recall, f1 directly from sklearn (should match your calculations)
    precision_class_wise, recall_class_wise, f1_class_wise, support = precision_recall_fscore_support(all_labels, all_preds, average=None)

    classes_to_include = [1,2,3,4,5,6,7,8]
    weighted_metrics_sklearn = precision_recall_fscore_support(all_labels, all_preds, labels=classes_to_include, average='weighted')
    weighted_precision = weighted_metrics_sklearn[0]
    weighted_recall = weighted_metrics_sklearn[1]
    weighted_f1 = weighted_metrics_sklearn[2]

    macro_f1 = precision_recall_fscore_support(all_labels, all_preds, average='macro')[2]

    print(f"Weighted Precision: {weighted_precision:.4f}, Weighted Recall: {weighted_recall:.4f}, Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}", flush=True)

    # Log to tensorboard only if writer is provided
    if writer is not None:
        writer.add_histogram('Metrics/Precision_per_class', precision_class_wise, epoch)
        writer.add_histogram('Metrics/Recall_per_class', recall_class_wise, epoch)
        writer.add_histogram('Metrics/F1_per_class', f1_class_wise, epoch)
        
        writer.add_scalar('Metrics/Weighted_Precision', weighted_precision, epoch)
        writer.add_scalar('Metrics/Weighted_Recall', weighted_recall, epoch)
        writer.add_scalar('Metrics/Weighted_F1', weighted_f1, epoch)
        writer.add_scalar('Metrics/Macro_F1', macro_f1, epoch)

        fig, ax = plt.subplots(figsize=(10, 8))
        class_names = ['Background', 'Brick_Furnace_1', 'Brick_Furnace_2', 'Brick_Furnace_3', 'Brick_Furnace_4', 'Brick_Furnace_5', 'Brick_Furnace_6', 'Brick_Furnace_7', 'Brick_Furnace_8']
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        
        plt.tight_layout()
        writer.add_figure('Metrics/Confusion_Matrix', fig, epoch)
        plt.close(fig)

    return weighted_f1  # Return the weighted F1 score for further use if needed
