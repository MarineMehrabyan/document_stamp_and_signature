import os
import glob
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib

SIZE = 224

train_dir = "signature_data/data"
real_images = []
forged_images = []

for per in os.listdir(train_dir):
    for data in glob.glob(os.path.join(train_dir, per, '*.*')):
        img = Image.open(data).convert('L')  # Convert to grayscale
        img = np.array(img)
        img = cv2.resize(img, (SIZE, SIZE))
        if per[-1] == 'g':
            forged_images.append(img)
        else:
            real_images.append(img)

real_images = np.array(real_images)
forged_images = np.array(forged_images)
real_labels = np.zeros((real_images.shape[0], 1))
forged_labels = np.ones((forged_images.shape[0], 1))
images = np.concatenate((real_images, forged_images))
labels = np.concatenate((real_labels, forged_labels))
images = images.reshape(images.shape[0], -1)

train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize and fit StandardScaler to the training data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)

# Save the fitted scaler using joblib.dump()
joblib.dump(scaler, 'signature_models/scaler.pkl')

def compute_hog_features(img):
    features = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    return features

hog_real_images = np.array([compute_hog_features(img) for img in real_images])
hog_forged_images = np.array([compute_hog_features(img) for img in forged_images])

hog_features = np.concatenate((hog_real_images, hog_forged_images))
hog_features = hog_features.reshape(hog_features.shape[0], -1)

hog_train_data, hog_test_data, hog_train_labels, hog_test_labels = train_test_split(
    hog_features, labels, test_size=0.2, random_state=42
)

hog_scaler = StandardScaler()
hog_train_data_scaled = hog_scaler.fit_transform(hog_train_data)

joblib.dump(hog_scaler, 'signature_models/hog_scaler.pkl')

knn_classifier = KNeighborsClassifier(n_neighbors=3)
poly_svm_classifier = svm.SVC(kernel='poly', C=10, degree=2, coef0=0.1, probability=True)

voting_classifier = VotingClassifier(
    estimators=[
        ('knn', knn_classifier),
        ('poly_svm', poly_svm_classifier),
    ],
    voting='soft'
)

voting_classifier.fit(hog_train_data_scaled, hog_train_labels.ravel())
hog_y_pred_voting = voting_classifier.predict(hog_scaler.transform(hog_test_data))

voting_accuracy = accuracy_score(hog_test_labels, hog_y_pred_voting)
voting_report = classification_report(hog_test_labels, hog_y_pred_voting)

print("Accuracy of Voting Classifier:", voting_accuracy)
print("Classification Report of Voting Classifier:\n", voting_report)

fpr_voting, tpr_voting, _ = roc_curve(hog_test_labels, hog_y_pred_voting)
roc_auc_voting = auc(fpr_voting, tpr_voting)

# Plot ROC curve
plt.figure(figsize=(10, 5))
plt.plot(fpr_voting, tpr_voting, color='blue', lw=2, label='ROC curve (Voting Classifier) (area = %0.2f)' % roc_auc_voting)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Voting Classifier)')
plt.legend(loc="lower right")
plt.show()

joblib.dump(voting_classifier, 'signature_models/voting_classifier_model.pkl')

