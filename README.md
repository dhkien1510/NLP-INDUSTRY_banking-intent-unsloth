# Banking Intent Classification with Unsloth Fine-tuning (PA02)

## 1. Project Description
This project fine-tunes the `gemma-2-2b-it` model to classify customer banking intents using the Unsloth framework. We extracted a subset of 30 core intents from the BANKING77 dataset. The data is split into Train (4158 samples), Validation (734 samples), and a perfectly balanced Test set of 1200 samples (exactly 40 samples per intent).

## 2. Model Performance
- **Evaluation Dataset**: 1200 samples (30 intents, 40 samples each).
- **Accuracy**: 92.50% (1110/1200 correct)
- **Macro F1-Score**: 0.80
- **Weighted F1-Score**: 0.93

### Error Analysis
The model performs exceptionally well (97-100% precision/recall) on discrete tasks like `card_arrival`, `cancel_transfer`, and `activate_my_card`. However, it shows slight confusion on status-related intents, specifically distinguishing between `pending_transfer` and `transfer_not_received_by_recipient` (Recall ~60%). This is expected as the semantic boundary between these natural language queries is extremely narrow.

## 3. Training Configurations & Hyperparameters

The model was fine-tuned using LoRA and 4-bit quantization via the Unsloth library. We utilized the **Completion-Only Training** technique (masking out user instructions) to compute the loss exclusively on the generated intent labels, preventing the model from memorizing the prompt.

### LoRA Configurations
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Target Modules**: Attention and MLP modules.

### Training Hyperparameters
- **Base Model**: `unsloth/gemma-2-2b-it-bnb-4bit`
- **Max Sequence Length**: 256 tokens
- **Batch Size**: 8 (per device)
- **Gradient Accumulation**: 4 (Effective batch size = 32)
- **Epochs**: 10
- **Learning Rate**: 2e-4
- **Optimizer**: `adamw_8bit`
- **LR Scheduler**: `cosine`
- **Warmup Ratio**: 0.1
- **Weight Decay**: 0.01

## 4. Setup Environment

To run this project, an NVIDIA GPU with CUDA installed is required due to Unsloth's native 4-bit loading.

```bash
# Navigate to the project directory
cd banking-intent-unsloth

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

## 5. How to Run

### Data Preprocessing
(Data is already placed in `sample_data/` folder).
```bash
python scripts/preprocess_data.py
```

### Training the Model
You can start training using the provided shell script. It reads hyperparameter configs from `configs/train.yaml`.
```bash
bash train.sh
# OR
python scripts/train_kaggle.py
```
*Note: The final LoRA checkpoint will be saved to `outputs/final_checkpoint`.*

### Running Inference
The inference script strictly implements the `IntentClassification` class interface. It initializes the model based on `configs/inference.yaml`.

**1. Single Query Classification:**
```bash
bash inference.sh query "My card was stolen, please block it immediately."
```

**2. Evaluate the Test Set:**
```bash
bash inference.sh test
```

**3. Interactive Mode:**
```bash
bash inference.sh
```
