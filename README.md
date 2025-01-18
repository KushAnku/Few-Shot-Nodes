# Virtual Node Tuning for Few-shot Node Classification

## Authors
- Jay Patel
- Shrey Patel
- Ankush Singh

## Abstract
This project tackles the challenge of few-shot node classification (FSNC) using sparse labeled data. We introduce **Virtual Node Tuning (VNT)**, an innovative approach leveraging a pre-trained graph transformer and meta-learning-enhanced pseudo prompts. The method demonstrates improved performance on four benchmark datasets, addressing challenges like sparse labeling and computational efficiency.

## Problem Statement
### Few-shot Node Classification (FSNC)
FSNC focuses on classifying nodes in symbolic graphs with limited labeled data per class. Traditional methods struggle with label scarcity and class imbalance. VNT addresses these issues using soft prompts and graph-based pseudo prompts for better information transfer.

### Application: Financial Fraud Detection
- **Background**: Fraud detection in financial systems suffers from rare labeled data and class imbalance.
- **Challenge**: Traditional methods require extensive labeled instances and struggle with computational inefficiencies.
- **Solution**: VNT improves fraud detection by:
  - Utilizing sparse labeled data efficiently.
  - Reducing computational overhead compared to retraining entire models.

## Methodology
### Virtual Node Tuning Overview
- A **Graph Transformer (GT)** is pre-trained as the encoder.
- **Virtual Nodes** act as soft prompts tailored to specific tasks.
- A **Graph-based Pseudo Prompt Evolution (GPPE)** module facilitates effective learning in sparse-label scenarios.

### Training Approach
1. **Data Preparation**:
   - Dataset: [Bitcoin Alpha trust weighted signed network](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html).
   - Features include User IDs, Ratings, and Timestamps.
2. **Model**: Two-layer **Graph Convolutional Network (GCN)**:
   - First layer: Transforms features into intermediate representations.
   - Second layer: Outputs logits for binary classification.
3. **Optimization**:
   - Loss: Binary Cross-Entropy with Logits.
   - Optimizer: Adam with a learning rate of 0.01.

### Meta-Learning (MAML)
A Model-Agnostic Meta-Learning (MAML) approach prepares the model for quick adaptation to new tasks with minimal data.

## Dataset
- **Source**: Bitcoin Alpha, a cryptocurrency trading platform.
- **Features**:
  - User IDs: Unique identifiers.
  - Ratings: Trust levels between users.
  - Timestamps: Temporal data.

## Code Snippets
### GCN Model
```python
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```
## Training Loop
The training loop optimizes the Graph Convolutional Network (GCN) model for few-shot node classification tasks.

```python
model = GCN(1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out.squeeze(), data.y.float())
    loss.backward()
    optimizer.step()
```
## Explanation

### Model
- The `GCN` class represents a simple two-layer Graph Convolutional Network.

### Optimizer
- The Adam optimizer is used with a learning rate of `0.01` for effective optimization.

### Loss Function
- Binary Cross-Entropy with Logits is chosen for binary classification tasks.

### Training Loop
- The model is set to training mode.
- Gradients are zeroed before performing a forward pass.
- The forward pass computes the predictions, and the loss is calculated.
- The backward pass updates the weights using gradient descent.

---

## Evaluation

### Metrics
The following metrics were used to evaluate the model's performance:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

---

### Visualization: Confusion Matrix
The confusion matrix is used to visualize the classification results. It provides insights into the types of errors made by the model.

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(val_labels, val_predictions)
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticklabels(['', 'Negative', 'Positive'])
ax.set_yticklabels(['', 'Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.show()
```

---

## References
- [Virtual Node Tuning for Few-shot Node Classification (arXiv)](https://arxiv.org/abs/2306.06063)
- [ACM Proceedings](https://dl.acm.org/doi/abs/10.1145/3580305.3599541)
- [YouTube Presentation on VNT](https://www.youtube.com/watch?v=vcaoAVBVPBQ)
- [Bitcoin Alpha Dataset](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)

