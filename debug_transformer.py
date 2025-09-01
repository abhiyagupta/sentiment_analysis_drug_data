#!/usr/bin/env python3
# debug_transformer.py - Diagnose transformer model issues

import os
import pickle
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def check_transformer_files():
    """Check what files exist in the transformer model directory"""
    model_dir = "./transformer_model"
    info_file = "transformer_model_info.pkl"
    
    print("🔍 TRANSFORMER MODEL DIAGNOSIS")
    print("=" * 50)
    
    # Check if directories/files exist
    print(f"📁 Model directory exists: {os.path.exists(model_dir)}")
    print(f"📄 Info file exists: {os.path.exists(info_file)}")
    
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        print(f"📋 Files in {model_dir}:")
        for f in sorted(files):
            file_path = os.path.join(model_dir, f)
            size = os.path.getsize(file_path) if os.path.isfile(file_path) else "DIR"
            print(f"   • {f} ({size} bytes)" if size != "DIR" else f"   • {f}/ (directory)")
    
    # Check info file contents
    if os.path.exists(info_file):
        try:
            with open(info_file, "rb") as f:
                info = pickle.load(f)
            print(f"\n📊 Info file contents:")
            for key, value in info.items():
                print(f"   • {key}: {value}")
        except Exception as e:
            print(f"❌ Error reading info file: {e}")
    
    print()

def test_transformer_loading():
    """Test different ways to load the transformer"""
    model_dir = "./transformer_model"
    
    print("🔧 TESTING TRANSFORMER LOADING METHODS")
    print("=" * 50)
    
    # Method 1: Direct pipeline from directory
    try:
        print("1. Testing pipeline from directory...")
        pipe1 = pipeline("sentiment-analysis", model=model_dir, return_all_scores=True)
        result1 = pipe1("This drug works great")
        print(f"   ✅ Success: {result1}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 2: Load model and tokenizer separately
    try:
        print("\n2. Testing separate model/tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        pipe2 = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)
        result2 = pipe2("This drug works great")
        print(f"   ✅ Success: {result2}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 3: Check if it's actually a fine-tuned model
    try:
        print("\n3. Testing as text-classification pipeline...")
        pipe3 = pipeline("text-classification", model=model_dir, return_all_scores=True)
        result3 = pipe3("This drug works great")
        print(f"   ✅ Success: {result3}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

def test_manual_inference():
    """Test manual inference without pipeline"""
    model_dir = "./transformer_model"
    
    print("🔬 TESTING MANUAL INFERENCE")
    print("=" * 50)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        text = "This drug works great"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        print(f"📊 Raw logits: {logits}")
        print(f"📊 Probabilities: {probabilities}")
        print(f"📊 Predicted class: {torch.argmax(probabilities, dim=-1).item()}")
        
        # Check model config for label mapping
        if hasattr(model.config, 'id2label'):
            print(f"📋 Model label mapping: {model.config.id2label}")
        
    except Exception as e:
        print(f"❌ Manual inference failed: {e}")

def compare_with_traditional():
    """Compare transformer vs traditional model on same input"""
    print("⚖️  COMPARING MODELS")
    print("=" * 50)
    
    test_text = "This drug works great"
    
    # Test traditional model
    try:
        import sys
        sys.path.append('.')
        from model import DrugSentimentPredictor
        
        print("Testing traditional model...")
        predictor_trad = DrugSentimentPredictor(prefer_transformer_if_available=False)
        trad_pred, trad_proba = predictor_trad.predict_single(test_text, "Aspirin")
        print(f"Traditional: {trad_pred} | Probabilities: {trad_proba}")
        
    except Exception as e:
        print(f"❌ Traditional model test failed: {e}")

if __name__ == "__main__":
    check_transformer_files()
    test_transformer_loading() 
    test_manual_inference()
    compare_with_traditional()
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Check your transformer training code - labels might be inverted")
    print("2. Verify your training data has correct sentiment labels") 
    print("3. Consider retraining the transformer with corrected labels")
    print("4. Use the traditional model (LGB/XGB) which seems to work correctly")