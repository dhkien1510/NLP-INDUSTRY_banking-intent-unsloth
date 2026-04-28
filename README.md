# Banking Intent Classification with Unsloth Fine-tuning (PA02)

## 1. Project Description
This project fine-tunes the `gemma-2-2b-it` model to classify customer banking intents using the Unsloth framework. We extracted a subset of 30 core intents from the BANKING77 dataset. The data is split into Train (4158 samples), Validation (734 samples), and a perfectly balanced Test set of 1200 samples (exactly 40 samples per intent).

## 2. Subset of Intents Used
We filtered the BANKING77 dataset to focus on the following 30 core intents:
`Refund_not_showing_up`, `activate_my_card`, `balance_not_updated_after_bank_transfer`, `balance_not_updated_after_cheque_or_cash_deposit`, `beneficiary_not_allowed`, `cancel_transfer`, `card_arrival`, `card_linking`, `card_payment_fee_charged`, `card_payment_not_recognised`, `card_payment_wrong_exchange_rate`, `cash_withdrawal_charge`, `cash_withdrawal_not_recognised`, `declined_card_payment`, `declined_cash_withdrawal`, `direct_debit_payment_not_recognised`, `extra_charge_on_statement`, `pending_card_payment`, `pending_cash_withdrawal`, `pending_top_up`, `pending_transfer`, `request_refund`, `reverted_card_payment?`, `top_up_failed`, `top_up_reverted`, `transaction_charged_twice`, `transfer_fee_charged`, `transfer_not_received_by_recipient`, `wrong_amount_of_cash_received`, `wrong_exchange_rate_for_cash_withdrawal`.

## 3. Model Performance
- **Evaluation Dataset**: 1200 samples (30 intents, exactly 40 samples each).
- **Accuracy**: 92.50% (1110/1200 correct)
- **Macro F1-Score**: 0.80
- **Weighted F1-Score**: 0.93

### Detailed Classification Report
```text
                                                  precision    recall  f1-score   support

                           Refund_not_showing_up       1.00      0.97      0.99        40
                                activate_my_card       1.00      0.95      0.97        40
         balance_not_updated_after_bank_transfer       0.81      0.85      0.83        40
balance_not_updated_after_cheque_or_cash_deposit       1.00      0.93      0.96        40
                         beneficiary_not_allowed       1.00      0.93      0.96        40
                                 cancel_transfer       1.00      1.00      1.00        40
                                    card_arrival       0.97      0.97      0.97        40
                                    card_linking       0.98      1.00      0.99        40
                      card_payment_charged_twice       0.00      0.00      0.00         0
                        card_payment_fee_charged       0.85      0.97      0.91        40
                       card_payment_not_accepted       0.00      0.00      0.00         0
                     card_payment_not_recognised       0.92      0.85      0.88        40
                card_payment_wrong_exchange_rate       0.97      0.97      0.97        40
                          cash_withdrawal_charge       0.95      0.90      0.92        40
                  cash_withdrawal_not_recognised       0.90      0.95      0.93        40
                           declined_card_payment       0.88      0.95      0.92        40
                           declined_cash_deposit       0.00      0.00      0.00         0
                        declined_cash_withdrawal       0.91      1.00      0.95        40
             direct_debit_payment_not_recognised       0.88      0.90      0.89        40
                       extra_charge_on_statement       1.00      0.95      0.97        40
                            pending_card_payment       0.92      0.90      0.91        40
                         pending_cash_withdrawal       0.97      0.97      0.97        40
                                  pending_top_up       0.88      0.95      0.92        40
                                pending_transfer       0.74      0.93      0.82        40
                                  request_refund       1.00      0.95      0.97        40
                       request_shipment_tracking       0.00      0.00      0.00         0
                          reverted_card_payment?       0.93      0.93      0.93        40
                                   top_up_failed       0.89      0.78      0.83        40
                                 top_up_reverted       0.80      0.90      0.85        40
                       transaction_charged_twice       0.98      1.00      0.99        40
                            transfer_fee_charged       1.00      0.95      0.97        40
              transfer_not_received_by_recipient       0.96      0.60      0.74        40
                   wrong_amount_of_cash_received       0.97      0.90      0.94        40
           wrong_cash_withdrawal_amount_received       0.00      0.00      0.00         0
         wrong_exchange_rate_for_cash_withdrawal       0.93      0.95      0.94        40

                                        accuracy                           0.93      1200
                                       macro avg       0.80      0.79      0.79      1200
                                    weighted avg       0.93      0.93      0.93      1200
```

### Error Analysis
The model performs exceptionally well (97-100% precision/recall) on discrete tasks like `card_arrival`, `cancel_transfer`, and `activate_my_card`. However, it shows slight confusion on status-related intents, specifically distinguishing between `pending_transfer` and `transfer_not_received_by_recipient` (Recall ~60%). This is expected as the semantic boundary between these natural language queries is extremely narrow. Note that Macro Avg F1-Score is 0.80 due to False Positives generated for zero-support (out-of-distribution) labels.

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
