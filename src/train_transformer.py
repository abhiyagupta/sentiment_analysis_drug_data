#!/usr/bin/env python3
# train_transformer_fixed.py

import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from data_module import DataModule

mlflow.set_experiment("Drug_Sentiment_Analysis_Transformer")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro, 
        "f1_weighted": f1_weighted
    }

def plot_performance(y_true, y_pred, target_f1, out_path="transformer_performance.png"):
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True)

    classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
    f1s = [report[c]["f1-score"] for c in classes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, f1s, color=["lightcoral", "lightblue", "lightgreen"])
    plt.axhline(target_f1, color="red", linestyle="--", label=f"Target {target_f1:.2f}")
    plt.axhline(f1, color="blue", linestyle="-", label=f"Overall Macro F1: {f1:.3f}")
    
    # Add value labels on bars
    for bar, f1_val in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{f1_val:.3f}', ha='center', va='bottom')
    
    plt.ylabel("F1-score")
    plt.title("Transformer Performance per Class")
    plt.legend()
    plt.ylim(0, max(max(f1s) + 0.1, target_f1 + 0.1))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\n📊 Creating transformer performance visualization...\n   Saved as: {out_path}")
    plt.close()

def analyze_predictions(y_true, y_pred, id2label):
    """Detailed analysis of predictions"""
    print("\n" + "="*60)
    print("DETAILED PREDICTION ANALYSIS")
    print("="*60)
    
    report = classification_report(y_true, y_pred, target_names=list(id2label.values()))
    print(report)
    
    # Confusion matrix analysis
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("True\\Pred", "\t".join(id2label.values()))
    for i, true_label in enumerate(id2label.values()):
        row_str = f"{true_label:<8}"
        for j in range(len(id2label)):
            row_str += f"\t{cm[i,j]}"
        print(row_str)

def main():
    dm = DataModule()
    target_f1 = 0.47

    if not os.path.exists("data\train.csv"):
        raise FileNotFoundError("train.csv not found")

    print("🔄 Loading and preprocessing data...")
    
    # Load + preprocess with cleaning
    df = dm.load_csv("data\train.csv")
    
    # Get initial data stats
    print("\n📊 Initial data statistics:")
    initial_stats = dm.get_data_stats(df)
    for key, value in initial_stats.items():
        print(f"  {key}: {value}")
    
    # Clean and prepare data
    df = dm.prepare_dataframe(df, clean_labels=True)
    
    # Get final data stats
    print("\n📊 Final data statistics:")
    final_stats = dm.get_data_stats(df)
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # Select only needed columns and rename
    df = df[["combined_text", "sentiment"]].rename(columns={"combined_text": "text"})
    
    # Verify we have clean data
    assert df['sentiment'].isna().sum() == 0, "Still have NaN sentiments!"
    assert len(df['sentiment'].unique()) <= 3, f"Too many sentiment classes: {df['sentiment'].unique()}"
    
    # Create consistent label mapping
    unique_sentiments = sorted(df["sentiment"].unique().tolist())
    print(f"\n🏷️  Found sentiment classes: {unique_sentiments}")
    
    # Standard mapping (regardless of what's in the data)
    label2id = {"positive": 0, "negative": 1, "neutral": 2}
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    
    # Map sentiments to IDs, handling missing classes
    df["label"] = df["sentiment"].map(label2id)
    
    # Check for unmapped labels
    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        print(f"⚠️  Warning: {unmapped} rows couldn't be mapped to standard labels")
        print("Unmapped sentiments:", df[df["label"].isna()]["sentiment"].unique())
        # Remove unmapped rows
        df = df.dropna(subset=["label"])
    
    df["label"] = df["label"].astype(int)
    print(f"✅ Label distribution: {df['label'].value_counts().sort_index().to_dict()}")

    # Split data
    print("\n🔪 Splitting data...")
    train_df, val_df = dm.train_val_split(df, stratify_col="sentiment")
    
    print(f"📊 Train set: {len(train_df)} samples")
    print(f"📊 Val set: {len(val_df)} samples")
    print(f"📊 Train label dist: {train_df['label'].value_counts().sort_index().to_dict()}")
    print(f"📊 Val label dist: {val_df['label'].value_counts().sort_index().to_dict()}")

    # Convert to Hugging Face datasets
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    # Model setup
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print(f"\n🤖 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenization function
    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
    
    print("🔤 Tokenizing datasets...")
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    # Set format for PyTorch
    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # Always 3 classes
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    # Training arguments with better settings for dirty data
    training_args = TrainingArguments(
        output_dir="./transformer_checkpoints",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps", 
        save_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        save_total_limit=2,
        seed=42,
    )

    # Log parameters to MLflow
    param_dict = {
        "model_name": model_name,
        "max_length": 128,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "epochs": training_args.num_train_epochs,
        "warmup_steps": training_args.warmup_steps,
        "weight_decay": training_args.weight_decay,
    }

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train model with MLflow tracking
    if mlflow.active_run():
        mlflow.end_run()
        
    with mlflow.start_run(run_name="transformer_roberta_cleaned"):
        # Log parameters
        for key, value in param_dict.items():
            mlflow.log_param(key, value)
        
        print("\n🚀 Starting training...")
        trainer.train()
        
        # Evaluate
        print("\n📊 Evaluating model...")
        metrics = trainer.evaluate()
        f1_macro = metrics.get("eval_f1_macro", 0)
        f1_micro = metrics.get("eval_f1_micro", 0)
        f1_weighted = metrics.get("eval_f1_weighted", 0)
        
        # Log metrics
        mlflow.log_metric("eval_f1_macro", f1_macro)
        mlflow.log_metric("eval_f1_micro", f1_micro)
        mlflow.log_metric("eval_f1_weighted", f1_weighted)
        
        print(f"\n=== Transformer (RoBERTa) Results ===")
        print(f"📊 Macro F1: {f1_macro:.4f}")
        print(f"📊 Micro F1: {f1_micro:.4f}")
        print(f"📊 Weighted F1: {f1_weighted:.4f}")
        
        status = "✅ PASS" if f1_macro >= target_f1 else "❌ FAIL"
        print(f"🎯 Target: {target_f1:.3f} | Status: {status}")

        # Get detailed predictions for analysis
        print("\n🔍 Generating predictions for analysis...")
        predictions = trainer.predict(val_ds)
        preds = np.argmax(predictions.predictions, axis=-1)
        y_true = val_df["label"].values
        y_pred = preds

        # Detailed analysis
        analyze_predictions(y_true, y_pred, id2label)
        
        # Plot performance
        plot_performance([id2label[i] for i in y_true],
                        [id2label[i] for i in y_pred],
                        target_f1)

        # Save model
        print("\n💾 Saving model...")
        model_save_path = "./transformer_model"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        # Save metadata
        info = {
            "model_name": model_save_path,
            "label_mapping": id2label,
            "f1_score": f1_macro,
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "epochs_trained": training_args.num_train_epochs,
        }
        
        info_path = "transformer_model_info.pkl"
        with open(info_path, "wb") as f:
            pickle.dump(info, f)

        # Final summary
        print("\n" + "="*60)
        print("TRANSFORMER TRAINING SUMMARY")
        print("="*60)
        print(f"Model: RoBERTa | Macro F1: {f1_macro:.4f} | {status}")
        print(f"🎯 Target F1-score: {target_f1}")
        print(f"🏆 Achieved: {f1_macro:.4f}")
        
        if f1_macro >= target_f1:
            print("🎉 SUCCESS! Model meets target performance.")
        else:
            print("⚠️  Model below target. Consider:")
            print("   • More training epochs")
            print("   • Data augmentation") 
            print("   • Different model architecture")
            print("   • Hyperparameter tuning")

        print(f"\n📁 Files created:")
        print(f"   • {model_save_path}/ (fine-tuned model + tokenizer)")
        print(f"   • {info_path} (metadata for inference)")
        print("   • transformer_performance.png (performance chart)")
        print("   • MLflow logs in ./mlruns/")
        
        print(f"\n▶️  Next step: Use model.py --predict or --test for inference")
        
        return f1_macro

if __name__ == "__main__":
    main()