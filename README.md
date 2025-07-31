import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load the file (note the delimiter is a semicolon)
file_path = r"C:\Users\fshos\Downloads\heart_cleaned.csv"
data = pd.read_csv(file_path, delimiter=';')

# Strip any whitespace from column names
data.columns = data.columns.str.strip()

# Print column names (for verification)
print("Columns:", data.columns.tolist())

# Check if 'target' column exists
if 'target' not in data.columns:
    raise ValueError("The column 'target' is not found in the dataset.")

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Store results
results = []

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    })
    print(f"\nModel: {model_name}")
    print(classification_report(y_test, y_pred))

# Display comparison table
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df)

# Function to extract one diseased and one healthy patient
def extract_patient_details(data):
    diseased_patient = data[data['target'] == 1].iloc[0].to_dict()
    healthy_patient = data[data['target'] == 0].iloc[0].to_dict()

    diseased_reasons = []
    if diseased_patient['thalach'] < 140:
        diseased_reasons.append("Low maximum heart rate (thalach).")
    if diseased_patient['thal'] in [1, 2]:
        diseased_reasons.append("Abnormal thalium stress test results (thal).")
    if diseased_patient['age'] >= 50:
        diseased_reasons.append("Older age (above 50 years).")

    healthy_reasons = []
    if healthy_patient['thalach'] >= 150:
        healthy_reasons.append("Good maximum heart rate (thalach).")
    if healthy_patient['thal'] == 3:
        healthy_reasons.append("Normal thalium stress test results (thal).")
    if healthy_patient['ca'] >= 1:
        healthy_reasons.append("Clear arteries (ca >= 1).")

    return {
        "Diseased_Patient": {
            "Details": diseased_patient,
            "Reasons": diseased_reasons
        },
        "Healthy_Patient": {
            "Details": healthy_patient,
            "Reasons": healthy_reasons
        }
    }

# Extract and print patient details
patient_details = extract_patient_details(data)

print("\nDiseased Patient Details:")
for key, value in patient_details["Diseased_Patient"]["Details"].items():
    print(f"{key}: {value}")
print("\nReasons for Disease:")
for reason in patient_details["Diseased_Patient"]["Reasons"]:
    print(f"- {reason}")

print("\nHealthy Patient Details:")
for key, value in patient_details["Healthy_Patient"]["Details"].items():
    print(f"{key}: {value}")
print("\nReasons for Health:")
for reason in patient_details["Healthy_Patient"]["Reasons"]:
    print(f"- {reason}")









