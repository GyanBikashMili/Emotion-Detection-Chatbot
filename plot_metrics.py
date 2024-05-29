import matplotlib.pyplot as plt

# Data from the classification report
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
precision = [0.43, 0.23, 0.36, 0.59, 0.38, 0.29, 0.64]
recall = [0.20, 0.24, 0.13, 0.75, 0.35, 0.52, 0.55]
f1_score = [0.27, 0.24, 0.19, 0.66, 0.37, 0.37, 0.59]

# Plot the data
plt.figure(figsize=(10, 6))

# Plot Precision
plt.plot(classes, precision, label='Precision', marker='o')

# Plot Recall
plt.plot(classes, recall, label='Recall', marker='o')

# Plot F1-Score
plt.plot(classes, f1_score, label='F1-Score', marker='o')

# Add titles and labels
plt.title('Classification Metrics for Each Emotion Class')
plt.xlabel('Emotion Classes')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
