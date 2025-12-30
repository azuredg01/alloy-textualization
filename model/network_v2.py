import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    get_scheduler,
    logging,
)

logging.set_verbosity_error()


class AlloyRegressor(torch.nn.Module):
    def __init__(self, config, backbone):
        super().__init__()
        self.backbone = backbone

        hidden_size = backbone.config.hidden_size  # 不用管是 BERT 還是 RoBERTa
        self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 只要是 BERT 類，通常都有 pooled_output
        pooled = outputs.pooler_output
        logits = self.head(pooled)
        return logits
    
    # ======= for attention visualization =======
    def forward_with_attn(self, input_ids, attention_mask=None):
        """
        額外提供一個同時輸出 prediction + attentions 的 forward
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        pooled = outputs.pooler_output
        logits = self.head(pooled)
        return logits, outputs.attentions

def create_model(config):
    model_name = config['model']['name_or_path']

    if config['stage'] == 'pretrain':
        # 用於 MLM 的 MatSciBERT / RoBERTa 等
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(config['device'])
    elif config['stage'] == 'finetune':
        backbone = AutoModel.from_pretrained(model_name)
        model = AlloyRegressor(config, backbone).to(config['device'])

    return model


def cri_opt_sch(config, model):
    if config['stage'] == 'pretrain':
        criterion = None
    elif config['stage'] == 'finetune':
        criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'])

    sch_cfg = config['sch']
    if sch_cfg.get('name', '').lower() == 'plateau':
        # ===== epoch-based：ReduceLROnPlateau =====
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = sch_cfg.get('mode', 'min'),      # loss/MAE/MSE→'min'；R2→'max'
            factor   = sch_cfg.get('factor', 0.5),
            patience = sch_cfg.get('patience', 3),
            verbose  = sch_cfg.get('verbose', True),
            threshold= sch_cfg.get('threshold', 0.0),   # 類似 min_delta
            threshold_mode = sch_cfg.get('threshold_mode', 'abs')
        )
    else:
        # ===== step-based：HuggingFace get_scheduler =====
        scheduler = get_scheduler(
            sch_cfg['name'],
            optimizer=optimizer,
            num_warmup_steps=sch_cfg.get('warmup_steps', 0),
            num_training_steps=int(config['train_len'] * config['epochs'])
        )

    return criterion, optimizer, scheduler
