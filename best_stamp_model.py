import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import numpy as np
import os
from PIL import Image
import cv2
import joblib  

SIZE = 224

base_dir = "stamp_data"
fake_dir = os.path.join(base_dir, "fake")
real_dir = os.path.join(base_dir, "real")

# Lists to store real and fake images
real_images = []
fake_images = []

def is_image(filename):
    try:
        with Image.open(filename) as img:
            img.verify()  
        return True
    except (IOError, OSError) as e:
        return False

for filename in os.listdir(fake_dir)[:100]: 
    img_path = os.path.join(fake_dir, filename)
    if is_image(img_path):
        try:
            img = Image.open(img_path).convert('L')  
            img = np.array(img)
            img = cv2.resize(img, (SIZE, SIZE))
            fake_images.append(img)
        except Exception as e:
            print(f"Skipping non-image file: {img_path}, Error: {e}")

for filename in os.listdir(real_dir)[:100]: 
    img_path = os.path.join(real_dir, filename)
    if is_image(img_path):
        try:
            img = Image.open(img_path).convert('L') 
            img = np.array(img)
            img = cv2.resize(img, (SIZE, SIZE))
            real_images.append(img)
        except Exception as e:
            print(f"Skipping non-image file: {img_path}, Error: {e}")

# onvert the lists of images into NumPy arrays 
real_images = np.array(real_images)
fake_images = np.array(fake_images)

real_labels = np.zeros((real_images.shape[0], 1))
fake_labels = np.ones((fake_images.shape[0], 1))
images = np.concatenate((real_images, fake_images))
labels = np.concatenate((real_labels, fake_labels))
images = images.reshape(images.shape[0], -1)

train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


smote = SMOTE(random_state=42)
train_data_resampled, train_labels_resampled = smote.fit_resample(train_data, train_labels.ravel())  # .ravel() to change shape from (n, 1) to (n,)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # binary classification
    eval_metric='logloss',        # metric to be used
    use_label_encoder=False       # as we're providing labels directly
)

param_grid = {
    'max_depth': [3, 6],                # depth of trees
    'n_estimators': [50, 100, 150],     # number of trees
    'learning_rate': [0.01, 0.1, 0.2]   # step size shrinkage used to prevent overfitting
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=3, verbose=1)  # Use 3-fold cross-validation
grid_search.fit(train_data_resampled, train_labels_resampled)

print("Best Parameters:", grid_search.best_params_)
#Best Parameters: {'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 100}
best_xgb_model = grid_search.best_estimator_

predictions = best_xgb_model.predict(test_data)

report = classification_report(test_labels, predictions)
print("Classification Report:\n", report)

joblib.dump(best_xgb_model, 'stamp_models/best_xgb_model.joblib')

