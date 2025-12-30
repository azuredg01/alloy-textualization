本專案用於將合金（Alloy）資料集轉換為可供 Transformer-based 模型
（如 RoBERTa、MatSciBERT）進行 **性質回歸（Regression）**
與 **結構分類（Classification）** 的訓練資料，
並透過 **HuggingFace Accelerate + Distributed Data Parallel (DDP)** 進行多 GPU 訓練。

## Pipeline
                                 Dataset
                               (CSV files)
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────┐
        │               Step 1: **data_process.ipynb**             │
        │               - 資料清理                                 │
        │               - 欄位命名統一（PROPERTY: ...）             │
        │               - 缺失值 / 異常值處理                       │
        └──────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
        ┌────────────────────────────┐ ┌────────────────────────────┐
        │ Step 2 (Regression)        │ │ Step 2 (Classification)    │
        │ **convertToPKL.py**        │ │ **convertToPKL_class.py**  │
        │ - Train/Val split          │ │ - Train/Val split          │
        │ - Target normalization     │ │ - Class label              │
        └────────────────────────────┘ └────────────────────────────┘
                    │                               │
                    │                               │
                    ▼                               ▼
        ┌────────────────────────────┐ ┌────────────────────────────┐
        │ Step 3 (Regression)        │ │ Step 3 (Classification)    │
        │ **generate_text.py**       │ │ **generate_text_class.**   │
        │ - Natural language text    │ │ - Natural language text    │
        │ - Remove target leakage    │ │ - Integer class target     │
        │ - target, target_raw       │ │ - target, target_raw       │
        └────────────────────────────┘ └────────────────────────────┘
                    │                               │
                    ▼                               ▼
            Train / Val CSV Files          Train / Val CSV Files
                    │                               │
                    └──────────────┬────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────────────┐
        │               Step 4: **config/*.yaml**                  │
        │               - Dataset paths (train / val pkl)          │
        │               - Model backbone                           │
        │               - Optimizer / Scheduler                    │
        │               - Metrics & Early Stopping                 │
        └──────────────────────────────────────────────────────────┘
                                     │
                     ┌───────────────┴──────────────────┐
                     │                                  │
                     ▼                                  ▼
        ┌────────────────────────────┐ ┌────────────────────────────┐
        │ Step 5 (Regression)        │ │ Step 5 (Classification)    │
        │ **run_regression.sh**      │ │ **run_class.sh**           │
        │ → main_DDP_v3.py           │ │ → main_DDP_class.py        │
        │ - MSE / MAE / R2           │ │ - Accuracy / AUC           │
        └────────────────────────────┘ └────────────────────────────┘
                                      │
                                      ▼
                              training_output/


## Step 1: Data Preprocessing

**Script :**
```
data_process.ipynb
```
- 清理原始合金資料
- 統一欄位格式（如 `PROPERTY: Microstructure`）
- 確保每一列資料對應一筆 alloy sample

```
output:

all_data/
└── all_data_xxx.csv
```


## Step 2: Dataset Conversion
```python
# 反開需要的 target:

target_dict = {
    'all_data_YS': 'PROPERTY: YS (MPa)',
    'all_data_YM': 'PROPERTY: Calculated Young modulus (GPa)',
    ...
}
```

### Regression :
```bash
python convertToPKL.py
```
- 固定 random seed 切分 train / val
- 僅使用 **training set** 的統計量做 target normalization

```
output:

all_data/<target>/
├── tr_<target>_convert.csv
└── vl_<target>_convert.csv
```

### Classification :
```bash
python convertToPKL_class.py
```
- 固定 random seed 切分 train / val
- 將類別（如 FCC / BCC / Other）轉成 label

```
output:

all_data/all_data_Micro/
├── tr_all_data_Micro_convert.csv
└── vl_all_data_Micro_convert.csv
```


## Step 3: Text Generation
```python
# 反開需要的 target:

target_dict = {
    'all_data_YS': 'PROPERTY: YS (MPa)',
    'all_data_YM': 'PROPERTY: Calculated Young modulus (GPa)',
    ...
}
```

### Regression :
```bash
python generate_text.py
```
### Classification :
```bash
python generate_text_class.py
```
**Output PKL fields**
- `text`
- `target`（normalized or class label）


## Step 4: Training Configuration
```
config/*.yaml
```
Each YAML defines:
- Dataset paths (`train_data`, `val_data`)
- Model backbone
- Optimizer & scheduler
- Metrics and early stopping

| Item | Regression | Classification |
|-----|-----------|----------------|
| Target | Continuous | Integer label |
| Loss | lr | lr |
| Metrics | MAE, MSE, R² | Sensitivity(Recall), Specificity, Accuracy, Auc |
| Monitor | val_R2 | val_auc / val_accuracy |

| 指標 | 公式 | 關注 |
| ----------- | ---------------- | --------------- |
| Accuracy    | (TP + TN) / All | 全體 |
| Precision   | TP / (TP + FP)  | FP |
| Sensitivity(Recall)      | TP / (TP + FN)  | FN   |
| Specificity | TN / (TN + FP)  | FP |
| F1 score | 2·TP / (2·TP + FP + FN) | FP + FN（平衡） |

## Accelerate GPU Configuration
```
gpu_yaml/default_config.yaml
```

This file is a **pure Accelerate configuration** and controls:
- Number of GPUs
- Distributed backend
- Mixed precision (if enabled)

Used by both:
- `run_regression.sh`
- `run_class.sh`


## Step 5: Model Training

### Regression
**Script :** ``main_DDP_v3.py``
```bash
bash run_regression.sh # 設定需依環境更改
```

### Classification
**Script :** ``main_DDP_class.py``
```bash
bash run_class.sh # 設定需依環境更改
```

### Training Outputs
#### For both regression and classification:
```
training_output/finetune-*/
├── config.yaml 
├── model.pt
├── ..
└── ..
```

---

### Notes

Regression 與 Classification 共同一條 Pipeline

差異僅在：

Step 2（ target 處理方式 ）

Step 3（ text generation script ）

Step 4（ training config 設定）

Step 5（ training script & run.sh ）

---

### Reference

AlloyBERT: Alloy Property Prediction with Large Language Models