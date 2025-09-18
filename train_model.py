import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# ============ Load Dataset ============
try:
    # Try reading with headers
    df = pd.read_csv("DATA.csv")
    print("✅ Loaded DATA.csv with headers.")
except Exception as e:
    print("⚠️ Error reading DATA.csv:", e)

# Drop extra unnamed columns (from Excel exports sometimes)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("🔎 Columns found in CSV:", df.columns.tolist())

# ============ Handle Headers ============
expected_cols = ['MQ4', 'MQ135', 'TEMPERATURE', 'PULSE', 'GAS CONDITION LEVEL']

if all(col in df.columns for col in expected_cols):
    print("✅ Using existing headers from CSV")
else:
    print("⚠️ No proper headers found. Assigning default column names.")
    df.columns = expected_cols

# ============ Features and Labels ============
X = df[['MQ4', 'MQ135', 'TEMPERATURE', 'PULSE']]
Y = df['GAS CONDITION LEVEL']

# Encode output labels (Normal / Precaution / Severe etc.)
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# ============ Train/test split ============
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ============ Feature scaling ============
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============ Train Model ============
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# ============ Save Model, Encoder, and Scaler ============
joblib.dump(model, "gastric_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained and pickle files saved successfully!")
