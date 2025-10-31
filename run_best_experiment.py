import numpy as np
from helpers import load_csv_data, create_csv_submission
from medical_dataset_helpers.preprocessing import MinMaxScaler
from medical_dataset_helpers.ensemble_models import RandomForestClassifier

print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("./data/dataset")

print("Cleaning NaN columns...")
temporary_stacked_x = np.vstack((x_train, x_test))
valid_cols = np.where(~np.isnan(temporary_stacked_x).any(axis=0))[0]
x_train_clean = x_train[:, valid_cols]
x_test_clean = x_test[:, valid_cols]

print(f"Cleaned shapes:")
print(f"  x_train_clean: {x_train_clean.shape}")
print(f"  x_test_clean: {x_test_clean.shape}")

print("Balancing dataset...")
y_train_binary = (y_train + 1) // 2  
n_positive = np.sum(y_train_binary == 1)
n_negative = np.sum(y_train_binary == 0)

positive_indices = np.where(y_train_binary == 1)[0]
negative_indices = np.where(y_train_binary == 0)[0]

n_oversample = n_negative // n_positive
oversampled_positive_indices = np.repeat(positive_indices, n_oversample)
balanced_indices = np.concatenate([negative_indices, oversampled_positive_indices])
np.random.seed(42)
np.random.shuffle(balanced_indices)

x_train_balanced = x_train_clean[balanced_indices]
y_train_balanced = y_train[balanced_indices]
y_train_binary_balanced = (y_train_balanced + 1) // 2

print(f"Balanced dataset:")
print(f"  Positive samples: {np.sum(y_train_binary_balanced == 1)}")
print(f"  Negative samples: {np.sum(y_train_binary_balanced == 0)}")
print(f"  Total samples: {len(x_train_balanced)}")

print("Applying MinMaxScaler...")
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train_balanced)
x_test_scaled = scaler.transform(x_test_clean)

print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    criterion='gini',
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf.fit(x_train_scaled, y_train_binary_balanced)
print("Predicting on test set...")
y_test_pred_binary = rf.predict(x_test_scaled)
y_test_pred = (y_test_pred_binary * 2) - 1

print("Creating submission file...")
create_csv_submission(test_ids, y_test_pred, "final_submission.csv")
print("Submission file 'final_submission.csv' created successfully!")

