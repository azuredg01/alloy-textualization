import torch
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    get_scheduler,
    logging,
)

logging.set_verbosity_error()


class AlloyClassifier(torch.nn.Module):
    """
    Fine-tune 用的分類模型：
    - backbone: BERT / RoBERTa 類
    - head: Linear(hidden_size → num_labels)
    """

    def __init__(self, config, backbone):
        super().__init__()
        self.backbone = backbone

        hidden_size = backbone.config.hidden_size

        if isinstance(config, dict):
            model_cfg = config.get("model", {}) or {}
            num_labels = int(model_cfg.get("num_labels", 1))
        else:
            num_labels = 1

        self.num_labels = num_labels
        self.head = torch.nn.Linear(hidden_size, self.num_labels) # 分類頭

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs.pooler_output
        logits = self.head(pooled)  # (B, num_labels)
        return logits

    def forward_with_attn(self, input_ids, attention_mask=None):
        """
        額外提供給 attention 視覺化使用
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        pooled = outputs.pooler_output
        logits = self.head(pooled)
        return logits, outputs.attentions


def create_model(config):
    model_name = config["model"]["name_or_path"]

    if config["stage"] == "pretrain":
        # MLM pretrain
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(config["device"])
    elif config["stage"] == "finetune":
        # 分類 fine-tune
        backbone = AutoModel.from_pretrained(model_name)
        model = AlloyClassifier(config, backbone).to(config["device"])
    else:
        raise ValueError(f"Unknown stage: {config['stage']}")

    return model


def cri_opt_sch(config, model):
    """
    - Pretrain: 不需要 criterion（由 transformers 處理）
    - Finetune: 一律視為分類 → CrossEntropyLoss
    """
    if config["stage"] == "pretrain":
        criterion = None
    elif config["stage"] == "finetune":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown stage: {config['stage']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optim"]["lr"])

    sch_cfg = config["sch"]
    if sch_cfg.get("name", "").lower() == "plateau":
        # epoch-based：ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch_cfg.get("mode", "min"),
            factor=sch_cfg.get("factor", 0.5),
            patience=sch_cfg.get("patience", 3),
            verbose=sch_cfg.get("verbose", True),
            threshold=sch_cfg.get("threshold", 0.0),
            threshold_mode=sch_cfg.get("threshold_mode", "abs"),
        )
    else:
        # step-based：HuggingFace get_scheduler
        scheduler = get_scheduler(
            sch_cfg["name"],
            optimizer=optimizer,
            num_warmup_steps=sch_cfg.get("warmup_steps", 0),
            num_training_steps=int(config["train_len"] * config["epochs"]),
        )

    return criterion, optimizer, scheduler
