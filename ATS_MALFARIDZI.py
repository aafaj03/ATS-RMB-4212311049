import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATASET_SIZE = 5000

print("Loading dataset...")
df = pd.read_csv('emnist-letters-train.csv', header=None)
labels = df.iloc[:, 0].values
images = df.iloc[:, 1:].values

print(f"Sampling {DATASET_SIZE} data...")
np.random.seed(42)
sampled_indices = []
samples_per_class = DATASET_SIZE // 26
for label in range(1, 27):
    label_indices = np.where(labels == label)[0]
    selected = np.random.choice(label_indices, samples_per_class, replace=False)
    sampled_indices.extend(selected)

X_images = images[sampled_indices].reshape(-1, 28, 28)
y_sampled = labels[sampled_indices]

print("Extracting HOG features...")
X_hog = []
for img in tqdm(X_images, desc="HOG"):
    img_fixed = np.fliplr(np.transpose(img))
    features = hog(img_fixed, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    X_hog.append(features)
X_hog = np.array(X_hog)

print(f"\nStarting LOOCV ({DATASET_SIZE} samples)...")
loo = LeaveOneOut()
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

y_true = []
y_pred = []

for train_idx, test_idx in tqdm(loo.split(X_hog), total=DATASET_SIZE, desc="LOOCV"):
    svm.fit(X_hog[train_idx], y_sampled[train_idx])
    pred = svm.predict(X_hog[test_idx])
    y_true.append(y_sampled[test_idx][0])
    y_pred.append(pred[0])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=range(1, 27))

print(f"\nResults:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score:  {f1:.4f}")

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[chr(64+i) for i in range(1, 27)],
            yticklabels=[chr(64+i) for i in range(1, 27)])
plt.title(f'Confusion Matrix - LOOCV\nAccuracy: {accuracy:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

print("\nFile saved: confusion_matrix.png")
print("Done!")
