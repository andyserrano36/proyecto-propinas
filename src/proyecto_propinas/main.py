import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error

# Cargar datos
tips = sns.load_dataset("tips")

# Variables
X = tips.drop(columns="tip")
y = tips["tip"]

# Identificar columnas
categoricas = ["sex", "smoker", "day", "time"]
numericas = ["total_bill", "size"]

# Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
    ]
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("modelo", LinearRegression())
    ]
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Entrenamiento
pipeline.fit(X_train, y_train)

# Predicción
y_pred = pipeline.predict(X_test)

# Evaluación
rmse = root_mean_squared_error(y_test, y_pred)

print("Te voy a mostrar el RMSE del modelo de regresión lineal que predice la propina")
print(f"RMSE: {rmse:.4f}")