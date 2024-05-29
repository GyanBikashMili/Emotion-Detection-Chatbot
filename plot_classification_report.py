import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data for class-wise metrics
report_data = {
    'Class': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
    'Precision': [0.43, 0.23, 0.36, 0.59, 0.38, 0.29, 0.64],
    'Recall': [0.20, 0.24, 0.13, 0.75, 0.35, 0.52, 0.55],
    'F1-Score': [0.27, 0.24, 0.19, 0.66, 0.37, 0.37, 0.59]
}

df_metrics = pd.DataFrame(report_data)

# Data for overall metrics
overall_metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [0.448203, 0.451770, 0.448203, 0.427880]
}

df_overall = pd.DataFrame(overall_metrics)

# Plot Precision, Recall, and F1-Score for each class
fig, axes = plt.subplots(4, 1, figsize=(10, 20))

sns.barplot(x='Precision', y='Class', data=df_metrics, ax=axes[0], palette='viridis')
axes[0].set_title('Precision for Each Class')
axes[0].set_xlim(0, 1)

sns.barplot(x='Recall', y='Class', data=df_metrics, ax=axes[1], palette='viridis')
axes[1].set_title('Recall for Each Class')
axes[1].set_xlim(0, 1)

sns.barplot(x='F1-Score', y='Class', data=df_metrics, ax=axes[2], palette='viridis')
axes[2].set_title('F1-Score for Each Class')
axes[2].set_xlim(0, 1)

sns.barplot(x='Metric', y='Value', data=df_overall, ax=axes[3], palette='viridis')
axes[3].set_title('Overall Model Performance Metrics')
axes[3].set_ylim(0, 1)

plt.tight_layout()
plt.show()
