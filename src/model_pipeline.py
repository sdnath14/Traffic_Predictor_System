import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

def train_and_save_model(data_path, model_path):
    df = pd.read_csv(data_path)

    features = ['Hour', 'DayOfWeekNum', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']
    target = 'TrafficSituationEncoded'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")

    
    le = LabelEncoder()
    le.fit(df['Traffic Situation'])
    label_map = dict(zip(le.transform(le.classes_), le.classes_))

    model_artifacts = {
        'model': model,
        'features': features,
        'label_map': label_map
    }
    joblib.dump(model_artifacts, model_path)

    return model