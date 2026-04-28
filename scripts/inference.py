"""
inference.py — Standalone inference script for the fine-tuned Banking Intent model.
Supports both Unsloth (GPU) and standard Transformers (CPU Fallback).
"""

import os
import argparse
import yaml
import pandas as pd
import torch

# Try to load Unsloth, but fallback to standard transformers if no GPU is found
try:
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    USE_UNSLOTH = True
except (ImportError, NotImplementedError):
    print("\n[WARNING] Unsloth requires a GPU, but none was found. Falling back to standard Transformers (CPU mode).")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    USE_UNSLOTH = False

class IntentClassification:
    def __init__(self, model_path: str):
        """
        Loads the configuration file, tokenizer, and model checkpoint.
        Note: As per requirements, `model_path` points to the YAML config file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Config file not found at {model_path}")
            
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rel_checkpoint_path = self.config.get("checkpoint_path", "outputs/final_checkpoint")
        self.checkpoint_dir = os.path.join(project_root, rel_checkpoint_path)
        
        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_dir}. Please train the model first.")

        self.max_seq_length = self.config.get("max_seq_length", 256)

        if USE_UNSLOTH:
            print(f"==== LOADING UNSLOTH MODEL (GPU) ====")
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.checkpoint_dir,
                max_seq_length=self.max_seq_length,
                load_in_4bit=True,
                dtype=None,
            )
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="gemma",
            )
            FastModel.for_inference(self.model)
        else:
            print(f"==== LOADING STANDARD TRANSFORMERS (CPU) ====")
            import json
            with open(os.path.join(self.checkpoint_dir, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "unsloth/gemma-2-2b-it-bnb-4bit")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="cpu",
                    load_in_4bit=False,
                    torch_dtype=torch.float32,
                )
            except Exception as e:
                print("\n[ERROR] Failed to load base model on CPU. Make sure you have enough RAM.")
                raise e

            self.model = PeftModel.from_pretrained(base_model, self.checkpoint_dir)
            self.model.eval()

        print("==== MODEL READY ====")

    def __call__(self, message: str) -> str:
        """
        Receives an input message and returns the predicted label.
        """
        messages = [
            {
                "role": "user",
                "content": f"Classify the banking intent of the following customer message. Respond with only the intent label.\n\nInput: {message}"
            }
        ]
        
        if USE_UNSLOTH:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
        else:
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=1.0,
                top_p=1.0,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return raw_output.split("\n")[0].strip()

    def evaluate_test_set(self, test_csv_path: str):
        if not os.path.exists(test_csv_path):
            print(f"Test data not found at {test_csv_path}")
            return
            
        print(f"\n==== EVALUATING ON TEST SET: {test_csv_path} ====")
        df_test = pd.read_csv(test_csv_path)
        df_test.dropna(subset=["text", "label_name"], inplace=True)
        
        predictions, ground_truths = [], []
        total = len(df_test)
        for idx, row in df_test.iterrows():
            if idx % 50 == 0:
                print(f"Processing {idx}/{total}...")
            predictions.append(self.__call__(row["text"]))
            ground_truths.append(row["label_name"])

        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        print(f"\nResults: {correct}/{total} correct — Accuracy: {correct/total*100:.2f}%")

        try:
            from sklearn.metrics import classification_report
            print("\n==== CLASSIFICATION REPORT ====")
            print(classification_report(ground_truths, predictions, zero_division=0))
        except ImportError:
            print("Install scikit-learn to see the detailed classification report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on the fine-tuned intent model.")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to inference config YAML.")
    parser.add_argument("--query", type=str, help="A single banking query to classify.")
    parser.add_argument("--eval_test", action="store_true", help="Run evaluation on the test dataset.")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_abs_path = os.path.join(project_root, args.config)
    
    classifier = IntentClassification(model_path=config_abs_path)
    
    if args.query:
        print(f"\nInput Message: {args.query}")
        print(f"Predicted Label: {classifier(message=args.query)}")
    elif args.eval_test:
        test_path = os.path.join(project_root, "sample_data", "test.csv")
        classifier.evaluate_test_set(test_csv_path=test_path)
    else:
        print("\n--- Interactive Inference Mode ---")
        while True:
            user_input = input("\nEnter banking message (or 'quit'): ")
            if user_input.lower() in ["quit", "exit"]: break
            if user_input.strip() == "": continue
            print(f"Predicted Label: {classifier(user_input)}")
