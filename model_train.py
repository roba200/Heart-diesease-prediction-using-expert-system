from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd

# Load your dataset
data = pd.read_csv('heart.csv')
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]  # Feature columns
y = data['target']  # Target column

# Train a decision tree classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Extract raw rules
raw_rules = export_text(clf, feature_names=list(X.columns))

# Function to make rules human-readable
def human_readable_rules(raw_rules):
    feature_mapping = {
        'age': 'Age',
        'sex': 'Gender (1 = Male, 0 = Female)',
        'cp': 'Chest Pain Type (0-3)',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar (>120 mg/dl, 1 = True, 0 = False)',
        'restecg': 'Resting ECG Results (0-2)',
        'thalach': 'Max Heart Rate Achieved',
        'exang': 'Exercise-Induced Angina (1 = Yes, 0 = No)',
        'oldpeak': 'ST Depression',
        'slope': 'Slope of ST Segment (0-2)',
        'ca': 'Number of Major Vessels (0-3)',
        'thal': 'Thalassemia (0-3)',
    }

    # Replace feature names with human-readable descriptions
    for feature, description in feature_mapping.items():
        raw_rules = raw_rules.replace(feature, description)
    
    return raw_rules

# Convert to human-readable rules
readable_rules = human_readable_rules(raw_rules)
print(readable_rules)