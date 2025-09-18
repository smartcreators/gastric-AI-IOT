import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# ============ Load Dataset ============
df = pd.read_csv("DATA.csv")

# Drop extra unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Show columns
print("🔎 Columns in CSV:", df.columns.tolist())

# ====== Use your real column names ======
X = df[['MQ4', 'MQ135', 'TEMPERATURE', 'PULSE']]
Y = df['GAS CONDITION LEVEL']   # this is your label column

# Encode output labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# Save model, encoder, and scaler
joblib.dump(model, "gastric_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained and .pkl files saved successfully!")