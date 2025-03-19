import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Load and clean data
df = pd.read_csv('C:/Users/DELL/Downloads/seattle-weather (1).csv')
df.dropna(inplace=True)

# Encode weather
le = LabelEncoder()
df['weather'] = le.fit_transform(df['weather'])

# Prepare features and labels
X = df[['temp_min', 'temp_max', 'precipitation', 'wind']]
y = df['weather']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# ✅ Save model in binary mode
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# ✅ Save encoder too
with open('encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Model and encoder saved successfully.")
