import matplotlib.pyplot as plt

# Data for plotting DT
models = [
    "DT, random split",
    "DT, temporal split"
]

scores = [
    [0.66, 0.68, 0.67, 0.67, 0.66],  # DT, with proxies, random split
    [0.61, 0.67, 0.69, 0.67, 0.69],  # DT, with proxies, temporal split
]

# Fold indices
folds = [1, 2, 3, 4, 5]

# Plotting
plt.figure(figsize=(10, 6))

# Plot each model's scores
for model, model_scores in zip(models, scores):
    plt.plot(folds, model_scores, marker='o', label=model)

# Add labels and title
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("F1 Scores per Fold for the DT")
plt.xticks(folds)
plt.legend(loc="lower right", fontsize=10)

# Add grid for better readability
plt.grid(alpha=0.5)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Data for plotting XGBoost
models = [
    "XGBoost, random split",
    "XGBoost, temporal split"
]

scores = [
    [0.63, 0.63, 0.61, 0.61, 0.64],  # XGBoost, random split
    [0.58, 0.61, 0.65, 0.65, 0.64]   # XGBoost, temporal split
]

# Fold indices
folds = [1, 2, 3, 4, 5]

# Plotting
plt.figure(figsize=(10, 6))

# Plot each model's scores
for model, model_scores in zip(models, scores):
    plt.plot(folds, model_scores, marker='o', label=model)

# Add labels and title
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("F1 Scores per Fold for XGBoost")
plt.xticks(folds)
plt.legend(loc="lower right", fontsize=10)

# Add grid for better readability
plt.grid(alpha=0.5)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Data for plotting Random Forest
models = [
    "Random Forest, random split",
    "Random Forest, temporal split"
]

scores = [
    [0.61897356, 0.62146893, 0.60842754, 0.61256545, 0.61834769],  # Random Forest, random split
    [0.58941345, 0.57775591, 0.64480874, 0.62961316, 0.62379421]   # Random Forest, temporal split
]

# Fold indices
folds = [1, 2, 3, 4, 5]

# Plotting
plt.figure(figsize=(10, 6))

# Plot each model's scores
for model, model_scores in zip(models, scores):
    plt.plot(folds, model_scores, marker='o', label=model)

# Add labels and title
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("F1 Scores per Fold for Random Forest")
plt.xticks(folds)
plt.legend(loc="lower right", fontsize=10)

# Add grid for better readability
plt.grid(alpha=0.5)

plt.tight_layout()
plt.show()
