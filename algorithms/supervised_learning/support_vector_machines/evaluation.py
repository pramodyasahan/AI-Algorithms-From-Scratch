import matplotlib.pyplot as plt

# Results Data
learning_rates = [0.001, 0.01, 0.1, 0.1]
lambdas = [0.01, 0.01, 0.01, 0.1]
accuracies = [0.47, 0.36, 0.61, 0.61]
precisions = [0.62, 0.62, 0.58, 0.59]
recalls = [0.47, 0.36, 0.61, 0.61]
f1_scores = [0.54, 0.46, 0.59, 0.60]

# Plotting Accuracy
plt.figure()
plt.plot(range(len(accuracies)), accuracies, marker='o', label='Accuracy')
plt.xticks(range(len(accuracies)), [f'lr={lr}, lambda={l}' for lr, l in zip(learning_rates, lambdas)], rotation=45)
plt.xlabel('Parameter Combination')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Parameter Combinations')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_plot.png')
plt.show()

# Plotting Precision
plt.figure()
plt.plot(range(len(precisions)), precisions, marker='o', label='Precision', color='orange')
plt.xticks(range(len(precisions)), [f'lr={lr}, lambda={l}' for lr, l in zip(learning_rates, lambdas)], rotation=45)
plt.xlabel('Parameter Combination')
plt.ylabel('Precision')
plt.title('Precision vs. Parameter Combinations')
plt.legend()
plt.tight_layout()
plt.savefig('precision_plot.png')
plt.show()

# Plotting Recall
plt.figure()
plt.plot(range(len(recalls)), recalls, marker='o', label='Recall', color='green')
plt.xticks(range(len(recalls)), [f'lr={lr}, lambda={l}' for lr, l in zip(learning_rates, lambdas)], rotation=45)
plt.xlabel('Parameter Combination')
plt.ylabel('Recall')
plt.title('Recall vs. Parameter Combinations')
plt.legend()
plt.tight_layout()
plt.savefig('recall_plot.png')
plt.show()

# Plotting F1-score
plt.figure()
plt.plot(range(len(f1_scores)), f1_scores, marker='o', label='F1-score', color='red')
plt.xticks(range(len(f1_scores)), [f'lr={lr}, lambda={l}' for lr, l in zip(learning_rates, lambdas)], rotation=45)
plt.xlabel('Parameter Combination')
plt.ylabel('F1-score')
plt.title('F1-score vs. Parameter Combinations')
plt.legend()
plt.tight_layout()
plt.savefig('f1score_plot.png')
plt.show()
