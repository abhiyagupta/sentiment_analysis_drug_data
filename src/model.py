# model.py
#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import f1_score, classification_report
from transformers import pipeline
 
# ensure nltk resources
for pkg in ("punkt", "stopwords", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

class DrugSentimentPredictor:
    def __init__(self, model_path='best_model.pkl', transformer_info_path='transformer_model_info.pkl', prefer_transformer_if_available=True):
        self.model_path = model_path
        self.transformer_info_path = transformer_info_path
        self.prefer_transformer = prefer_transformer_if_available
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.model_type = "traditional"  # or 'transformer'
        self.transformer_pipeline = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        self._load()

    def _load(self):
        # Try transformer first if available and preferred
        transformer_model_dir = "./transformer_model"
        if self.prefer_transformer and os.path.exists(transformer_model_dir):
            try:
                # Load from the actual saved model directory, not just the info pickle
                self.model_type = "transformer"
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model=transformer_model_dir,
                    tokenizer=transformer_model_dir,
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Try to load label mapping if info file exists
                if os.path.exists(self.transformer_info_path):
                    with open(self.transformer_info_path, "rb") as f:
                        info = pickle.load(f)
                    self.transformer_label_map = info.get("label_mapping", None)
                    print(f"Loaded transformer model: {transformer_model_dir} (F1: {info.get('f1_score', 'unknown')})")
                else:
                    self.transformer_label_map = None
                    print(f"Loaded transformer model: {transformer_model_dir}")
                return
            except Exception as e:
                print("Transformer load failed; falling back to traditional:", e)

        # Load traditional pipeline
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.model_path} not found; run train.py first")

        with open(self.model_path, "rb") as f:
            pipeline_data = pickle.load(f)

        self.model = pipeline_data["model"]
        self.vectorizer = pipeline_data["vectorizer"]
        self.label_encoder = pipeline_data["label_encoder"]
        self.model_type = "traditional"
        # Build easy mapping from encoded -> string label
        classes = list(self.label_encoder.classes_)
        # label_encoder.transform gives ints in range [0..len-1]; inverse_transform maps back
        self.encoded_to_label = {i: classes[i] for i in range(len(classes))}
        print(f"Loaded traditional model: {type(self.model).__name__}")
        print(f"Label classes: {classes}")

    def preprocess_text(self, text):
        if pd.isna(text) or text == "":
            return ""
        t = str(text).lower()
        t = re.sub(r"http\S+|www\S+|https\S+|\S+@\S+|<.*?>", "", t)
        t = re.sub(r"[^a-zA-Z\s!?.,']", " ", t)
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
        tokens = word_tokenize(t)
        processed = [self.lemmatizer.lemmatize(tok) for tok in tokens if tok not in self.stop_words and len(tok) > 1]
        return " ".join(processed)

    def _map_label_to_readable(self, label_str):
        """Turn label like 'positive' or 'pos' into 'Positive' for final output."""
        if label_str is None:
            return "neutral"
        s = str(label_str).lower()
        if s.startswith("pos"):
            return "positive"
        if s.startswith("neg"):
            return "negative"
        if s.startswith("neu"):
            return "neutral"
        # fallback: return original lowercased
        return s

    def predict_single(self, text, drug="unknown"):
        if self.model_type == "transformer":
            return self._predict_transformer(text)
        return self._predict_traditional(text, drug)

    def _predict_transformer(self, text):
        try:
            # The pipeline returns a list of lists when return_all_scores=True
            results = self.transformer_pipeline(str(text)[:512])
            
            # Handle the nested structure correctly
            if isinstance(results, list) and len(results) > 0:
                scores_list = results[0]
            else:
                scores_list = results
            
            # Convert list of dicts to dict for easier access
            scores = {}
            for score_dict in scores_list:
                scores[score_dict['label']] = score_dict['score']
            
            # Find the label with highest score
            best_label = max(scores.items(), key=lambda kv: kv[1])[0]
            
            # The transformer is returning direct labels: 'negative', 'neutral', 'positive'
            # Use these directly and map probabilities correctly
            sentiment = best_label.lower()
            
            # Return probabilities in order: [positive, negative, neutral]
            proba_array = np.array([
                scores.get('positive', 0),   # positive
                scores.get('negative', 0),   # negative
                scores.get('neutral', 0)     # neutral
            ])
            
            return sentiment, proba_array
            
        except Exception as e:
            print("Transformer inference failed:", e)
            import traceback
            traceback.print_exc()
            return "neutral", np.array([0.33, 0.33, 0.34])

    def _predict_traditional(self, text, drug):
        tmp = pd.DataFrame({"text":[text], "drug":[drug]})
        tmp["clean_text"] = tmp["text"].apply(self.preprocess_text)
        tmp["combined_text"] = tmp["clean_text"] + " " + tmp["drug"].astype(str).str.lower()
        X = self.vectorizer.transform(tmp["combined_text"])
        pred_encoded = int(self.model.predict(X)[0])
        pred_proba = self.model.predict_proba(X)[0] if hasattr(self.model, "predict_proba") else None
        original_label = self.encoded_to_label.get(pred_encoded, None)  # e.g. 'positive'
        readable = self._map_label_to_readable(original_label)
        return readable, pred_proba

    def predict_batch(self, input_csv, output_csv="submission.csv", id_col="unique_hash"):
        df = pd.read_csv(input_csv)
        if "text" not in df.columns:
            raise ValueError("input must contain 'text' column")

        # If no unique_hash, create numeric IDs
        if id_col not in df.columns:
            df[id_col] = range(len(df))

        if "drug" not in df.columns:
            df["drug"] = "unknown"

        preds = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            pred, _ = self.predict_single(row["text"], row.get("drug", "unknown"))
            preds.append(pred)

        # Rename unique_hash → id
        out = pd.DataFrame({"id": df[id_col], "sentiment": preds})
        out.to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")
        return out
    
    def compare_models(self, csv_path="None"):
        """Compare transformer vs classical on validation split"""
        
        # Default path
        if csv_path is None:
            csv_path = os.path.join("data", "train.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")

        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "sentiment" not in df.columns:
            raise ValueError("train.csv must have 'text' and 'sentiment'")

        # Take 20% as validation for fair compare
        from sklearn.model_selection import train_test_split
        val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])[1]

        y_true = val_df["sentiment"].tolist()
        preds_trad, preds_trans = [], []

        # classical
        if self.model is not None:
            for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Classical"):
                p, _ = self._predict_traditional(row["text"], row.get("drug", "unknown"))
                preds_trad.append(p)

        # transformer
        if self.transformer_pipeline is not None:
            for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Transformer"):
                p, _ = self._predict_transformer(row["text"])
                preds_trans.append(p)

        print("\n📊 Comparison Report")
        if preds_trad:
            f1_trad = f1_score(y_true, preds_trad, average="macro")
            print(f"Traditional ({type(self.model).__name__}): Macro-F1={f1_trad:.4f}")
            print(classification_report(y_true, preds_trad))
        if preds_trans:
            f1_trans = f1_score(y_true, preds_trans, average="macro")
            print(f"Transformer: Macro-F1={f1_trans:.4f}")
            print(classification_report(y_true, preds_trans))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="CSV with text column for batch prediction")
    parser.add_argument("--predict", help="single text to predict")
    parser.add_argument("--drug", default="unknown")
    parser.add_argument("--compare", action="store_true", help="run both models on validation split")
    parser.add_argument("--sanity", action="store_true", help="run sanity check on obvious examples")
    args = parser.parse_args()

    predictor = DrugSentimentPredictor()

    if args.sanity:
        predictor.test_model_sanity()
        return

    if args.predict:
        sentiment, proba = predictor.predict_single(args.predict, args.drug)
        print(f"Input: {args.predict}")
        print(f"Drug: {args.drug}")
        print(f"Predicted Sentiment: {sentiment}")

        # Extra human-readable label mapping
        label_map = {
            0: "Positive",
            1: "Negative", 
            2: "Neutral",
            "0": "Positive",
            "1": "Negative",
            "2": "Neutral"
        }
        if sentiment in label_map:
            print(f"Sentiment label: {label_map[sentiment]}")

        if proba is not None:
            print("Probabilities:", proba)
        return
        

    if args.test:
        predictor.predict_batch(args.test)
        return

    if args.compare:
        predictor.compare_models()
        return


if __name__ == "__main__":
    main()