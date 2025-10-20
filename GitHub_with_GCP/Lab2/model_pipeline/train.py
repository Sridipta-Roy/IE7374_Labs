from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


from data_pipeline.data_fetcher import fetch_data
from .save import bump_version, save_model



def main():
    df = fetch_data()
    X = df.drop(columns=["species"]) # target column provided by sklearn iris
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)


    version = bump_version()
    artifact_path = save_model(model, version)


    print(f"trained_version={version}")
    print(f"accuracy={acc:.4f}")
    print(f"artifact={artifact_path}")

if __name__ == "__main__":
    main()