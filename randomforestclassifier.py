import pandas as pd
import numpy as np
import warnings
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")  # Suppress warnings for clean output
# -------------------------
# Step 1: Load CSV dataset
# -------------------------
csv_path = "combined.csv"
df = pd.read_csv(csv_path)

# -------------------------
# Step 2: Clean and preprocess
# -------------------------
df['text'] = df['Text'].astype(str)
df['text_len'] = df['text'].apply(len)
df['page_num'] = df['Page']
df['label'] = df['Level'].astype(str)

print("📄 Columns in DataFrame:", df.columns.tolist())

# Drop unused columns
cols_to_drop = [col for col in ['Text', 'Level', 'PDF Filename', 'Page'] if col in df.columns]
df = df.drop(columns=cols_to_drop)

# -------------------------
# Step 3: Encode target and filter rare classes
# -------------------------
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

min_samples_required = 4
valid_labels = df['label_encoded'].value_counts()[lambda x: x >= min_samples_required].index
df = df[df['label_encoded'].isin(valid_labels)]

# -------------------------
# Step 4: Define features and handle NaNs
# -------------------------
X = df[['text', 'text_len', 'page_num', 'Font Size', 'Bold', 'Italic', 'Line Spacing']]
y = df['label_encoded']

X['Font Size'] = X['Font Size'].fillna(X['Font Size'].median())
X['Bold'] = X['Bold'].fillna(0)
X['Italic'] = X['Italic'].fillna(0)
X['Line Spacing'] = X['Line Spacing'].fillna(X['Line Spacing'].median())
X['text'] = X['text'].fillna("")

print("❗ Any NaNs remaining?", X.isna().sum().sum())

# -------------------------
# Step 5: Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Step 6: Pipeline with SMOTE
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2)), 'text'),
        ('num', StandardScaler(), ['text_len', 'page_num', 'Font Size', 'Bold', 'Italic', 'Line Spacing'])
    ]
)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=2)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    ))
])

# -------------------------
# Step 7: Train and evaluate
# -------------------------
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))

present_labels = np.unique(y_test)
class_names = le.inverse_transform(present_labels)
print("✅ Classification Report:\n", classification_report(
    y_test, y_pred, labels=present_labels, target_names=class_names
))

# -------------------------
# Step 8: Save model and encoder
# -------------------------
joblib.dump(pipeline, "rf_pipeline_tuned.pkl")
joblib.dump(le, "label_encoder_tuned.pkl")
print("✅ Model and label encoder saved.")

# -------------------------
# Step 9: JSON Output (Predictions)
# -------------------------
# Predict on full dataset (not just test set)
all_preds = pipeline.predict(X)
df['Predicted_Level'] = le.inverse_transform(all_preds)

json_output = []
for i, row in df.iterrows():
    json_output.append({
        "Document": row.get("Document", ""),
        "Page": int(row.get("page_num", 0)),
        "Text": row['text'],
        "Predicted_Level": row['Predicted_Level']
    })

with open("predicted_output.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=4, ensure_ascii=False)

print("✅ JSON saved to predicted_output.json")

