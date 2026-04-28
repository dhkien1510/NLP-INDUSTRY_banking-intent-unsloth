from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import re
import os
import json


# 1. Load dataset
print("==== READ DATA ====")
dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)

df_train = dataset['train'].to_pandas()
df_test = dataset['test'].to_pandas()

if not df_train.empty and not df_test.empty:
    print("Read successfully")

# 2. Clean dataset
# Clean duplicate data
print("==== CLEAN DATA ====")
print(f"Dataset before: Train: {len(df_train)} Test: {len(df_test)}")
df_train.drop_duplicates(inplace=True)
df_test.drop_duplicates(inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)
print(f"Dataset after remove duplicate and NaN: Train: {len(df_train)} Test: {len(df_test)}")

# 3. EDA dataset
# Mapping to feature name

# Label distribution - graph
def draw_plot(data, fig_name, title_name, x_title, y_title):
    plt.figure(figsize=(10, 8))
    bar = plt.bar(data.index, data.values)
    max_idx = data.values.argmax()
    max_val = data.values[max_idx]
    min_idx = data.values.argmin()
    min_val = data.values[min_idx]
    print(max_val, min_val)

    bar[max_idx].set_color('red')
    bar[min_idx].set_color('red')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.text(data.index[min_idx], min_val, str(min_val), ha='center', va='bottom', color='red', fontweight='bold')
    plt.text(data.index[max_idx], max_val, str(max_val), ha='center', va='bottom', color='red', fontweight='bold')

    plt.title(title_name)
    plt.savefig(fig_name)

train_count = df_train['label'].value_counts()
test_count = df_test['label'].value_counts()

draw_plot(train_count, "label_distribution_train_barplot", "Label Distribution in Train Set", "Label", "Number")
draw_plot(test_count, "label_distribution_test_barplot", "Label Distribution in Train Set", "Label", "Number")

# Label distribution - table
print("=== Label Distribution by Table ===")
print(df_train['label'].value_counts())
print(df_test['label'].value_counts())

# Text length distribution - graph
text_len = {}
for text in df_train['text']:
    if len(text) not in text_len.keys():
        text_len[len(text)] = 1
    else:
        text_len[len(text)] += 1

plt.figure(figsize=(10, 8))
plt.bar(text_len.keys(), text_len.values())
plt.savefig("text_length_distribution")

# Text length box plot
print(df_train['text'].apply(len).quantile(0.95))



plt.figure(figsize=(10,8))
sns.boxplot(y=df_train['text'].apply(len))
plt.title("Boxplot phân phối độ dài câu")
plt.ylabel("Độ dài (số ký tự)")
plt.savefig("text_len_distribution_box_plot")
# N-gram

# Lấy danh sách văn bản cần đếm từ tập train
texts = df_train['text'].dropna().tolist()

# 1. & 2. Cấu hình CountVectorizer để tạo Bigram 
vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')

# Trích xuất đặc trưng và đếm
X = vectorizer.fit_transform(texts)

# 3. Tính tổng tần suất của mỗi Bigram trên toàn bộ tập văn bản
frequencies = sum(X).toarray()[0]
# Lấy danh sách các câu từ ghép tương ứng
ngrams = vectorizer.get_feature_names_out()

# Tạo Dictionary map giữa cụm n-gram và số lần xuất hiện
ngram_freq_dict = dict(zip(ngrams, frequencies))

# (Tùy chọn) Sắp xếp và chỉ lấy 100 cụm từ phổ biến nhất để Word Cloud không bị quá rối
top_ngrams = dict(sorted(ngram_freq_dict.items(), key=lambda item: item[1], reverse=True)[:100])

# 4. Tạo WordCloud từ dictionary tần suất
wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    colormap='viridis'  # Đổi bảng màu nếu muốn
).generate_from_frequencies(top_ngrams)

# 5. Trực quan hóa bằng matplotlib
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') # Tắt hiển thị trục toạ độ
plt.title("Bigram Word Cloud (Top 100 cụm 2 từ phổ biến nhất)", fontsize=16)
plt.savefig("wordcloud_bigram.png")

# ============================================================
# 4. Label mapping (integer -> intent name)
# ============================================================
print("\n==== LABEL MAPPING ====")
label_names = dataset['train'].features['label'].names
label_map = {i: name for i, name in enumerate(label_names)}

df_train['label_name'] = df_train['label'].map(label_map)
df_test['label_name'] = df_test['label'].map(label_map)

for k, v in list(label_map.items())[:10]:
    print(f"  {k}: {v}")
print(f"  ... ({len(label_map)} total labels)")

# ============================================================
# 5. Text normalization
# ============================================================
print("\n==== TEXT NORMALIZATION ====")

def normalize_text(text):
    """Lowercase, strip whitespace, collapse multiple spaces."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

df_train['text'] = df_train['text'].apply(normalize_text)
df_test['text'] = df_test['text'].apply(normalize_text)

print("Applied: lowercase, strip, collapse whitespace")
print(f"Sample: '{df_train['text'].iloc[0]}'")

# ============================================================
# 6. Create subset data — select a subset of intents
# ============================================================
print("\n==== SUBSET SELECTION ====")

NUM_SELECTED_LABELS = 30   # Use 30 out of 77 intents (adjustable)
RANDOM_SEED = 42

# Pick the top-N most frequent labels to keep a balanced subset
selected_labels = (
    df_train['label']
    .value_counts()
    .head(NUM_SELECTED_LABELS)
    .index
    .tolist()
)
selected_label_names = [label_map[l] for l in selected_labels]

print(f"Selected {NUM_SELECTED_LABELS} / {len(label_map)} labels")
print(f"Examples: {selected_label_names[:5]} ...")

# Filter train & test to only keep rows with selected labels
df_train_sub = df_train[df_train['label'].isin(selected_labels)].copy()
df_test_sub  = df_test[df_test['label'].isin(selected_labels)].copy()

print(f"Subset sizes — Train: {len(df_train_sub)}, Test: {len(df_test_sub)}")

# ============================================================
# 7. Re-map labels to contiguous [0, K) range
# ============================================================
print("\n==== LABEL RE-MAPPING ====")

old_to_new  = {old: new for new, old in enumerate(sorted(selected_labels))}
new_to_name = {new: label_map[old] for old, new in old_to_new.items()}

df_train_sub['original_label'] = df_train_sub['label']
df_test_sub['original_label']  = df_test_sub['label']

df_train_sub['label'] = df_train_sub['original_label'].map(old_to_new)
df_test_sub['label']  = df_test_sub['original_label'].map(old_to_new)

# Update label_name to match new mapping
df_train_sub['label_name'] = df_train_sub['label'].map(new_to_name)
df_test_sub['label_name']  = df_test_sub['label'].map(new_to_name)

print(f"Re-mapped to [0, {NUM_SELECTED_LABELS})")
for old, new in list(old_to_new.items())[:5]:
    print(f"  {label_map[old]} (was {old}) -> {new}")

# ============================================================
# 8. Train / Validation split
# ============================================================
print("\n==== TRAIN / VAL SPLIT ====")

VAL_RATIO = 0.15

df_train_final, df_val = train_test_split(
    df_train_sub,
    test_size=VAL_RATIO,
    random_state=RANDOM_SEED,
    stratify=df_train_sub['label'],
)

print(f"Train: {len(df_train_final)}, Val: {len(df_val)}, Test: {len(df_test_sub)}")

# ============================================================
# 9. Save processed data to sample_data/
# ============================================================
print("\n==== SAVING DATA ====")

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
os.makedirs(output_dir, exist_ok=True)

save_cols = ['text', 'label', 'label_name']

df_train_final[save_cols].to_csv(os.path.join(output_dir, "train.csv"), index=False)
df_val[save_cols].to_csv(os.path.join(output_dir, "val.csv"), index=False)
df_test_sub[save_cols].to_csv(os.path.join(output_dir, "test.csv"), index=False)

# Save label mapping JSON (useful for training & inference)
label_mapping = {
    "num_labels": NUM_SELECTED_LABELS,
    "id2label": {str(k): v for k, v in new_to_name.items()},
    "label2id": {v: k for k, v in new_to_name.items()},
}

with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=2, ensure_ascii=False)

print(f"Saved to: {output_dir}")
print(f"  train.csv  — {len(df_train_final)} samples")
print(f"  val.csv    — {len(df_val)} samples")
print(f"  test.csv   — {len(df_test_sub)} samples")
print(f"  label_mapping.json — {NUM_SELECTED_LABELS} labels")
print("\n==== DONE ====")