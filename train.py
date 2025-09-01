#!/usr/bin/env python3
# train.py

import os
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from data_module import DataModule

mlflow.set_experiment("Drug_Sentiment_Analysis_Training")

class Trainer:
    def __init__(self, target_f1=0.47, seed=42):
        self.seed = seed
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_score = 0.0
        self.best_name = None
        self.target_f1 = target_f1

    def create_tfidf(self, df, max_features=8000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words='english',
            sublinear_tf=True,
            norm='l2'
        )
        X = self.vectorizer.fit_transform(df['combined_text'])

        # Log TF-IDF params to MLflow
        params = self.vectorizer.get_params()
        for p, v in params.items():
            try:
                mlflow.log_param(f"tfidf_{p}", v)
            except Exception:
                mlflow.log_param(f"tfidf_{p}", str(v))

        return X

    def train_traditional(self, X, y):
        models = {
            "LightGBM": lgb.LGBMClassifier(
                random_state=self.seed, 
                class_weight='balanced', 
                n_estimators=150,
                verbosity=-1  # Silences LightGBM 
                ),
            "XGBoost": xgb.XGBClassifier(
                random_state=self.seed, 
                n_estimators=150, 
                use_label_encoder=False, 
                eval_metric='mlogloss',
                verbosity=0  # Silences XGBoost
                ),
            "Random Forest": RandomForestClassifier(
                random_state=self.seed, 
                n_estimators=150, 
                class_weight='balanced', 
                n_jobs=-1,
                verbose=0  # Silences Random Forest
                )
        }

        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        for name, model in models.items():
            print(f"\n=== {name} ===")

            # Ensure no run is left open
            if mlflow.active_run():
                mlflow.end_run()
            with mlflow.start_run(run_name=f"{name}_CV"):
                try:
                    if name == "XGBoost":
                        X_cv = X.toarray() if hasattr(X, "toarray") and X.shape[1] < 10000 else X
                        cv_scores = []
                        for train_idx, val_idx in tqdm(cv.split(X_cv, y), total=cv.get_n_splits(), desc=f"{name} CV"):
                            model.fit(X_cv[train_idx], y[train_idx])
                            preds = model.predict(X_cv[val_idx])
                            cv_scores.append(f1_score(y[val_idx], preds, average='macro'))
                    else:
                        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)

                    mean_score = float(np.mean(cv_scores))
                    std_score = float(np.std(cv_scores))

                    # Train final model on full data
                    if name == "XGBoost":
                        model.fit(X_cv, y)
                    else:
                        model.fit(X, y)

                    mlflow.log_param("model_name", name)
                    # log all hyperparameters of the model
                    params = model.get_params()
                    for p, v in params.items():
                        try:
                            mlflow.log_param(p, v)
                        except Exception:
                            mlflow.log_param(p, str(v))
                    mlflow.log_metric("cv_f1_macro_mean", mean_score)
                    mlflow.log_metric("cv_f1_macro_std", std_score)
                    mlflow.sklearn.log_model(model, artifact_path=name.lower())

                    # Pretty print
                    print(f"📊 Macro F1: {mean_score:.4f} (±{std_score:.4f})")
                    print(f"📈 Individual scores: {[f'{s:.3f}' for s in cv_scores]}")
                    if mean_score >= self.target_f1:
                        print(f"✅ Meets target ({self.target_f1:.3f})")
                    else:
                        print(f"⚠️  Below target ({self.target_f1:.3f})")

                    results[name] = {"model": model, "cv_mean": mean_score, "cv_std": std_score, "cv_scores": cv_scores}
                except Exception as e:
                    print(f"❌ Error training {name}: {e}")
                    continue

        if results:
            best_name = max(results.keys(), key=lambda n: results[n]["cv_mean"])
            self.best_model = results[best_name]["model"]
            self.best_score = results[best_name]["cv_mean"]
            self.best_name = best_name
            print(f"\n🏆 BEST MODEL: {best_name}")
            print(f"   Macro F1: {self.best_score:.4f}")
            return results
        else:
            raise RuntimeError("No model trained successfully")

    def save_pipeline(self, out_path="best_model.pkl"):
        pipeline = {
            "model": self.best_model,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder
        }
        with open(out_path, "wb") as f:
            pickle.dump(pipeline, f)
        print("\n💾 Model pipeline saved to:", out_path)
        print("   This includes:")
        print("   ✓ Trained model")
        print("   ✓ TF-IDF vectorizer")
        print("   ✓ Label encoder")
        print("   ✓ Preprocessing parameters")
        return out_path


def plot_results(results, target_f1=0.47):
    names = list(results.keys())
    means = [results[n]["cv_mean"] for n in names]
    stds = [results[n]["cv_std"] for n in names]

    plt.figure(figsize=(8,5))
    plt.bar(names, means, yerr=stds, capsize=6, color=['skyblue','lightgreen','salmon'])
    plt.axhline(target_f1, color='red', linestyle='--', label=f"Target {target_f1:.2f}")
    plt.ylabel("Macro F1 (mean ± std)")
    plt.title("Model Comparison")
    plt.legend()
    plt.savefig("model_comparison.png", bbox_inches="tight")
    print("\n📊 Creating performance visualizations...")
    print("   Saved as: model_comparison.png")
    plt.close()


def main():
    dm = DataModule()
    trainer = Trainer()

    if not os.path.exists("train.csv"):
        raise FileNotFoundError("train.csv not found in current directory")

    df = dm.load_csv("train.csv")
    df = dm.prepare_dataframe(df)

    # encode labels
    y = trainer.label_encoder.fit_transform(df["sentiment"].values)
    X = trainer.create_tfidf(df)

    # train traditional models
    results = trainer.train_traditional(X, y)

    # Save best
    saved_path = trainer.save_pipeline("best_model.pkl")

    # Plot results
    plot_results(results, trainer.target_f1)

    # Summary
    print("\n============================================================")
    print("TRAINING SUMMARY")
    print("============================================================")
    for name, res in results.items():
        status = "✅ PASS" if res["cv_mean"] >= trainer.target_f1 else "❌ FAIL"
        print(f"{name:<13} | F1: {res['cv_mean']:.4f} (±{res['cv_std']:.4f}) | {status}")

    print("\n🎯 Target F1-score:", trainer.target_f1)
    print(f"🏆 Best achieved: {trainer.best_score:.4f} ({trainer.best_name})")
    if trainer.best_score >= trainer.target_f1:
        print("🎉 SUCCESS! Ready for submission.")
    else:
        print("⚠️  Did not reach target — consider tuning or transformer training.")

    print("\n📁 Files created:")
    print("   • best_model.pkl (trained pipeline)")
    print("   • model_comparison.png (performance chart)")
    print("   • MLflow logs in ./mlruns/")

    print("\n▶️  Next step: Use model.py to make predictions")


if __name__ == "__main__":
    main()
