"""
inference.py — Standalone inference script for the fine-tuned Banking Intent model using Unsloth.
"""

import os
import argparse
import yaml
import pandas as pd
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

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
            
        # Get paths relative to the project root (assuming script runs from project root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Resolve checkpoint path from config
        rel_checkpoint_path = self.config.get("checkpoint_path", "outputs/final_checkpoint")
        self.checkpoint_dir = os.path.join(project_root, rel_checkpoint_path)
        
        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_dir}. Please train the model first.")

        self.max_seq_length = self.config.get("max_seq_length", 256)

        print(f"==== LOADING UNSLOTH MODEL FROM {self.checkpoint_dir} ====")
        
        # Load the model using Unsloth as required
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.checkpoint_dir,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True, # Unsloth 4-bit loading (Requires GPU)
            dtype=None,
        )

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma",
        )
        
        FastModel.for_inference(self.model)
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
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=48,
            temperature=1.0,
            top_p=1.0,
            do_sample=False,
            use_cache=True,
        )

        # Extract only the newly generated tokens
        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return raw_output.split("\n")[0].strip()

    def evaluate_test_set(self, test_csv_path: str):
        """
        Evaluates the model on a test CSV dataset and prints the classification report.
        """
        if not os.path.exists(test_csv_path):
            print(f"Test data not found at {test_csv_path}")
            return
            
        print(f"\n==== EVALUATING ON TEST SET: {test_csv_path} ====")
        df_test = pd.read_csv(test_csv_path)
        df_test.dropna(subset=["text", "label_name"], inplace=True)
        
        predictions = []
        ground_truths = []
        
        # Optional: Add a progress counter
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
            report = classification_report(ground_truths, predictions, zero_division=0)
            print(report)
        except ImportError:
            print("Install scikit-learn to see the detailed classification report.")


# ============================================================
# Short usage example showing how the inference class is called
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on the fine-tuned intent model.")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to inference config YAML.")
    parser.add_argument("--query", type=str, help="A single banking query to classify.")
    parser.add_argument("--eval_test", action="store_true", help="Run evaluation on the test dataset.")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_abs_path = os.path.join(project_root, args.config)
    
    # 1. Initialize the inference class using the config path (as requested by instruction)
    classifier = IntentClassification(model_path=config_abs_path)
    
    # 2. Inference a single text input
    if args.query:
        print(f"\nInput Message: {args.query}")
        predicted_label = classifier(message=args.query)
        print(f"Predicted Label: {predicted_label}")
        
    # 3. Evaluate the test set
    elif args.eval_test:
        test_path = os.path.join(project_root, "sample_data", "test.csv")
        classifier.evaluate_test_set(test_csv_path=test_path)
        
    # 4. Interactive mode (if no args provided)
    else:
        print("\n--- Interactive Inference Mode ---")
        print("Example usage of the IntentClassification class.")
        print("Type 'quit' to exit.")
        while True:
            user_input = input("\nEnter banking message: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if user_input.strip() == "": continue
            
            # Call the class instance directly
            pred = classifier(user_input)
            print(f"Predicted Label: {pred}")
