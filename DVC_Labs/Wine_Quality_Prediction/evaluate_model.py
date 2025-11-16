"""
Model Evaluation Module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib
import json
import os

def evaluate_model():
    """Generate evaluation plots and detailed metrics"""
    print("Starting model evaluation...")
    
    # Load model
    model = joblib.load('models/model.pkl')
    
    # Load test data
    test_data = pd.read_csv('data/features/test_features.csv')
    X_test = test_data.drop('quality_label', axis=1)
    y_test = test_data['quality_label']
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.close()
    print("✓ Confusion matrix saved")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png', dpi=150)
    plt.close()
    print("✓ ROC curve saved")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=150)
    plt.close()
    print("✓ Feature importance plot saved")
    
    # Prediction Distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(y_proba[y_test == 0], bins=30, alpha=0.7, label='Not Good Wine', color='red')
    plt.hist(y_proba[y_test == 1], bins=30, alpha=0.7, label='Good Wine', color='green')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Distribution by True Label')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(y_proba)), y_proba, c=y_test, cmap='RdYlGn', alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability')
    plt.title('Prediction Probabilities')
    plt.colorbar(label='True Label')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/prediction_distribution.png', dpi=150)
    plt.close()
    print("✓ Prediction distribution saved")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save detailed report
    with open('metrics/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save feature importance
    feature_importance.to_csv('metrics/feature_importance.csv', index=False)
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Not Good', 'Good Wine']))
    
    print("\n✓ Evaluation completed!")
    print(f"  - Plots saved in: plots/")
    print(f"  - Metrics saved in: metrics/")

if __name__ == '__main__':
    evaluate_model()