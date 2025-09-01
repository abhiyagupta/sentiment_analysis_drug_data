# Preprocessing :  data_module.py
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data available (safe to call repeatedly)
for pkg in ("punkt", "stopwords", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

class DataModule:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def load_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        df = pd.read_csv(path)
        return df

    def clean_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dirty sentiment labels to standard format
        """
        df = df.copy()
        
        # Convert sentiment to string first to handle mixed types
        df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
        
        # Map various representations to standard labels
        sentiment_mapping = {
            # Numeric representations
            '0': 'positive',
            '0.0': 'positive', 
            '1': 'negative',
            '1.0': 'negative',
            '2': 'neutral',
            '2.0': 'neutral',
            
            # Text representations
            'negative': 'negative',
            'neg': 'negative',
            'bad': 'negative',
            'poor': 'negative',
            'awful': 'negative',
            'terrible': 'negative',
            'hate': 'negative',
            'worst': 'negative',
            'horrible': 'negative',
            
            'neutral': 'neutral',
            'neut': 'neutral',
            'ok': 'neutral',
            'okay': 'neutral',
            'average': 'neutral',
            'fair': 'neutral',
            'mixed': 'neutral',
            
            'positive': 'positive',
            'pos': 'positive',
            'good': 'positive',
            'great': 'positive',
            'excellent': 'positive',
            'amazing': 'positive',
            'love': 'positive',
            'best': 'positive',
            'wonderful': 'positive',
            'fantastic': 'positive',
            
            # Handle blanks/nulls/invalid
            'nan': 'neutral',
            'none': 'neutral',
            '': 'neutral',
            ' ': 'neutral',
        }
        
        # Apply mapping
        df['sentiment'] = df['sentiment'].map(sentiment_mapping)
        
        # For any unmapped values, try to infer or set to neutral
        unmapped_mask = df['sentiment'].isna()
        if unmapped_mask.any():
            print(f"Warning: Found {unmapped_mask.sum()} unmapped sentiment values. Setting to 'neutral'.")
            print("Sample unmapped values:", df.loc[unmapped_mask, 'sentiment'].head().tolist())
            df.loc[unmapped_mask, 'sentiment'] = 'neutral'
        
        print(f"Sentiment distribution after cleaning:")
        print(df['sentiment'].value_counts())
        
        return df

    def basic_validate(self, df: pd.DataFrame) -> None:
        required = ["text", "sentiment"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def preprocess_text(self, text):
        if pd.isna(text) or text == "":
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+|\S+@\S+|<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s!?.,']", " ", text)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # reduce repeated chars
        tokens = word_tokenize(text)
        processed = []
        for t in tokens:
            if t not in self.stop_words and len(t) > 1:
                processed.append(self.lemmatizer.lemmatize(t))
        return " ".join(processed)

    def prepare_dataframe(self, df: pd.DataFrame, fill_drug_unknown=True, clean_labels=None) -> pd.DataFrame:
        self.basic_validate(df)
        df = df.copy()
        
        # Clean sentiment labels first if requested
        # Auto-detect if cleaning is needed when clean_labels=None
        if clean_labels is None:
            # Check if we have dirty labels that need cleaning
            sentiment_str = df['sentiment'].astype(str).str.strip().str.lower()
            standard_values = {'0', '1', '2', 'positive', 'negative', 'neutral', 'pos', 'neg', 'neut'}
            has_dirty_labels = not sentiment_str.isin(standard_values | {'nan', 'none', ''}).all()
            clean_labels = has_dirty_labels or df['sentiment'].isna().any()
            
        if clean_labels:
            df = self.clean_sentiment_labels(df)
        
        # Clean text
        df["text"] = df["text"].fillna("").astype(str)
        
        # Handle drug column
        if fill_drug_unknown:
            if "drug" not in df.columns:
                df["drug"] = "unknown"
            else:
                df["drug"] = df["drug"].fillna("unknown").astype(str)
        
        # Process text
        df["clean_text"] = df["text"].apply(self.preprocess_text)
        df["drug"] = df["drug"].astype(str).str.lower()
        df["combined_text"] = df["clean_text"] + " " + df["drug"]
        
        # Final validation - remove any rows with missing essential data
        initial_len = len(df)
        df = df.dropna(subset=['sentiment', 'text'])
        df = df[df['text'].str.strip() != '']  # Remove empty text
        final_len = len(df)
        
        if initial_len != final_len:
            print(f"Removed {initial_len - final_len} rows with missing/invalid data")
        
        return df

    def train_val_split(self, df: pd.DataFrame, test_size=0.2, stratify_col="sentiment"):
        if stratify_col not in df.columns:
            return train_test_split(df, test_size=test_size, random_state=self.seed)
        
        # Check if we have enough samples per class for stratification
        min_samples = df[stratify_col].value_counts().min()
        if min_samples < 2:
            print(f"Warning: Some classes have only {min_samples} samples. Cannot stratify. Using random split.")
            return train_test_split(df, test_size=test_size, random_state=self.seed)
        
        return train_test_split(
            df, test_size=test_size, random_state=self.seed, stratify=df[stratify_col]
        )

    def get_data_stats(self, df: pd.DataFrame) -> dict:
        """Get comprehensive statistics about the dataset"""
        stats = {
            'total_rows': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'empty_text_count': (df['text'].str.strip() == '').sum(),
            'missing_sentiment': df['sentiment'].isna().sum(),
        }
        
        if 'drug' in df.columns:
            stats['unique_drugs'] = df['drug'].nunique()
            stats['most_common_drugs'] = df['drug'].value_counts().head().to_dict()
        
        return stats