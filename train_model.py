import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TITANIC SURVIVAL PREDICTION - TRAINING")
print("="*60)

# 1. Load Data
print("\n[1] Loading data...")
try:
    df = pd.read_csv('data/train.csv')
    print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ ไม่พบไฟล์ data/train.csv")
    exit()

# 2. Data Preprocessing
print("\n[2] Data Preprocessing...")
df_processed = df.copy()

# เลือก features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df_processed = df_processed[features + ['Survived']].copy()

# จัดการ Missing Values - แก้ไขแล้ว
print("   Handling missing values...")

# Age - เติมด้วยค่าเฉลี่ย
age_imputer = SimpleImputer(strategy='mean')
df_processed['Age'] = age_imputer.fit_transform(df_processed[['Age']])

# Embarked - เติมด้วยค่าที่พบบ่อยที่สุด
# ตรวจสอบว่ามี missing values หรือไม่
if df_processed['Embarked'].isnull().sum() > 0:
    print(f"   Found {df_processed['Embarked'].isnull().sum()} missing values in Embarked")
    # แปลงเป็น string ก่อน
    df_processed['Embarked'] = df_processed['Embarked'].astype(str)
    embarked_imputer = SimpleImputer(strategy='most_frequent')
    df_processed['Embarked'] = embarked_imputer.fit_transform(df_processed[['Embarked']])
else:
    print("   No missing values in Embarked")
    embarked_imputer = SimpleImputer(strategy='most_frequent')
    embarked_imputer.fit(df_processed[['Embarked']])

# Encoding
print("   Encoding categorical variables...")
df_processed['Sex'] = df_processed['Sex'].map({'male': 0, 'female': 1})
df_processed = pd.get_dummies(df_processed, columns=['Embarked'], prefix='Embarked', drop_first=False)

# แยก X, y
X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Preprocessing complete")

# 3. Train Models
print("\n[3] Training models...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes': GaussianNB()
}

results = []

for name, model in models.items():
    print(f"\n🔹 {name}:")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Model_Object': model
    })

results_df = pd.DataFrame(results)

# 4. Save Best Model
best_idx = results_df['Accuracy'].idxmax()
best_model_name = results_df.iloc[best_idx]['Model']
best_model = results_df.iloc[best_idx]['Model_Object']

print("\n" + "="*60)
print(f"✅ Best Model: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[best_idx]['Accuracy']:.4f}")
print("="*60)

# Save files
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump({
    'age_imputer': age_imputer,
    'embarked_imputer': embarked_imputer,
    'feature_names': list(X.columns)
}, 'preprocessors.pkl')

print("\n✓ Saved: best_model.pkl, scaler.pkl, preprocessors.pkl")
print("\n Training complete!")