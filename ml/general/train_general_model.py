# train_general_model.py (Përditësuar për BCSC Risk Factors Dataset)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib, os
import numpy as np

# Llogarit root-in e projektit dinamikisht (për path-e relative të sakta)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))  # Ngjitet në root (nga ml/general/ në root)

# Siguro që ekziston folderi models në root
models_dir = os.path.join(root_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# 1️⃣ Lexo datasetin BCSC Risk Factors (supozo se e ke shpëtuar si CSV në data/general/bcsc_risk_factors.csv)
# Bashko të tre CSV-të nëse i ke të ndarë: df1 = pd.read_csv(..._1.csv); df = pd.concat([df1, df2, df3])
data_path = os.path.join(root_dir, "data", "general", "general.csv")
df = pd.read_csv(data_path)

# Shënim: Dataset-i është i agreguar me 'count' – do ta përdorim si sample_weight për trajnim
print(f"Dataset shape: {df.shape}")
print(df.head())  # Shiko të dhënat e para

# 2️⃣ Pastro dhe përgatit features (hiq kolonat e panevojshme si 'year' dhe 'count' nga X)
# Features kryesore: age_group_5_years, race_eth, first_degree_hx, age_menarche, age_first_birth,
# BIRADS_breast_density, current_hrt, menopaus, bmi_group, biophx
feature_cols = [
    'age_group_5_years', 'race_eth', 'first_degree_hx', 'age_menarche', 'age_first_birth',
    'BIRADS_breast_density', 'current_hrt', 'menopaus', 'bmi_group', 'biophx'
]
X = df[feature_cols]
y = df['breast_cancer_history']  # Target: 0/1 për histori kanceri gjiri

# Përdor 'count' si sample_weight për të marrë parasysh frekuencën
sample_weights = df['count']

# 3️⃣ Ndaje X dhe y (me stratify për balancim, duke pasur parasysh madhësinë e madhe të dataset-it)
# Për dataset të madh, sample 10% për shpejtësi (hiqe nëse do full)
df_sample = df.sample(frac=0.1, random_state=42)  # Opsionale: Zëvendëso me df nëse do të plotin
X = df_sample[feature_cols]
y = df_sample['breast_cancer_history']
sample_weights = df_sample['count']

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Trajnim i modelit (me StandardScaler dhe sample_weight)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')  # 'balanced' për të trajtuar klasat e pabalancuara
model.fit(X_train, y_train, sample_weight=weights_train)

# 5️⃣ Raporti dhe ruajtja
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
model_path = os.path.join(models_dir, "bcsc_general_model.joblib")
joblib.dump({
    "model": model, 
    "scaler": scaler, 
    "features": feature_cols
}, model_path)
print(f"✅ Saved to {model_path}")