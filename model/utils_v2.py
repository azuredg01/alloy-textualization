# utils_DDP.py  —— Drop-in replacement (Accelerate-safe)
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

# ---------------- helpers ----------------
def _get_attn(batch):
    """Robustly fetch attention_mask without tensor boolean ops."""
    attn = batch.get('mask', None) if isinstance(batch, dict) else None
    if attn is None and isinstance(batch, dict):
        attn = batch.get('attention_mask', None)
    return attn

def _to(x, device):
    return x.to(device, non_blocking=True)

def _tqdm(iterable, acc=None):
    """Disable progress bars on non-main ranks."""
    disable = (acc is not None) and (not acc.is_main_process)
    return tqdm(iterable, disable=disable)

def _forward_with_mask(model, ids, attention_mask=None):
    """Be tolerant to model signatures and outputs."""
    if attention_mask is None:
        out = model(ids)
    else:
        try:
            out = model(input_ids=ids, attention_mask=attention_mask)
        except TypeError:
            out = model(ids, attention_mask=attention_mask)
    if hasattr(out, "logits"):
        out = out.logits
    return out

# ---------------- pretrain ----------------
def train_pt(model, dataloader, optimizer, scheduler, device, acc=None):
    model.train()
    total_loss, lr_sum = 0.0, 0.0

    for batch in _tqdm(dataloader, acc):
        ids     = _to(batch['ids'], device)
        attn    = _to(batch['mask'], device) if 'mask' in batch else _to(batch['attention_mask'], device)
        labels  = _to(batch['labels'], device)

        optimizer.zero_grad(set_to_none=True)

        out  = model(ids, attention_mask=attn, labels=labels)
        loss = out.loss

        if acc is not None:
            acc.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        lr_sum     += optimizer.param_groups[0]['lr']

    return (total_loss / max(1, len(dataloader))), (lr_sum / max(1, len(dataloader)))


@torch.no_grad()
def validate_pt(model, dataloader, device, acc=None):
    model.eval()
    total_loss = 0.0

    for batch in _tqdm(dataloader, acc):
        ids     = _to(batch['ids'], device)
        attn    = _to(batch['mask'], device) if 'mask' in batch else _to(batch['attention_mask'], device)
        labels  = _to(batch['labels'], device)

        out  = model(ids, attention_mask=attn, labels=labels)
        loss = out.loss
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))

# ---------------- finetune ----------------
def train_ft(model, dataloader, optimizer, criterion, scheduler, device, acc=None):
    model.train()
    total_loss, lr_sum = 0.0, 0.0

    for batch in _tqdm(dataloader, acc):
        if isinstance(batch, dict):
            ids    = _to(batch['ids'], device)
            attn   = _get_attn(batch)
            attn   = _to(attn, device) if attn is not None else None
            target = _to(batch['target'], device).float()
        else:
            # support tuple style (ids, [attn], target)
            if len(batch) == 3:
                ids, attn, target = batch
                ids    = _to(ids, device)
                attn   = _to(attn, device) if attn is not None else None
                target = _to(target, device).float()
            else:
                ids, target = batch
                ids    = _to(ids, device)
                target = _to(target, device).float()
                attn   = None

        optimizer.zero_grad(set_to_none=True)

        out  = _forward_with_mask(model, ids, attn).squeeze(-1)
        loss = criterion(out, target)

        if acc is not None:
            acc.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        # 只有 step-based 才在 batch 內 step；Plateau 讓 main 在每個 epoch 後 step(...)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        total_loss += loss.item()
        lr_sum     += optimizer.param_groups[0]['lr']

    return (total_loss / max(1, len(dataloader))), (lr_sum / max(1, len(dataloader)))


@torch.no_grad()
def validate_ft(model, val_loader, criterion, device, acc=None):
    model.eval()

    # batch-avg stats (match your loss display)
    total_loss = 0.0
    total_mae  = 0.0
    n_batches  = 0

    # true MSE over all samples
    sum_sq_error = 0.0
    n_samples    = 0

    # R2 collections
    y_true_all = []
    y_pred_all = []

    for batch in _tqdm(val_loader, acc):
        if isinstance(batch, dict):
            ids    = _to(batch['ids'], device)
            attn   = _get_attn(batch)
            attn   = _to(attn, device) if attn is not None else None
            target = _to(batch['target'], device).float()
        else:
            if len(batch) == 3:
                ids, attn, target = batch
                ids    = _to(ids, device)
                attn   = _to(attn, device) if attn is not None else None
                target = _to(target, device).float()
            else:
                ids, target = batch
                ids    = _to(ids, device)
                target = _to(target, device).float()
                attn   = None

        pred = _forward_with_mask(model, ids, attn).squeeze(-1)
        loss = criterion(pred, target)
        mae  = torch.mean(torch.abs(target - pred))

        total_loss += loss.item()
        total_mae  += mae.item()
        n_batches  += 1

        diff = (pred - target)
        sum_sq_error += torch.sum(diff * diff).item()
        n_samples    += target.numel()

        if acc is not None:
            pred_g   = acc.gather_for_metrics(pred)
            target_g = acc.gather_for_metrics(target)
        else:
            pred_g, target_g = pred, target

        y_true_all.append(target_g.detach().cpu())
        y_pred_all.append(pred_g.detach().cpu())

    val_loss = total_loss / max(1, n_batches)
    val_mae  = total_mae  / max(1, n_batches)
    val_mse  = (sum_sq_error / n_samples) if n_samples > 0 else float('nan')

    if y_true_all:
        y_true_np = torch.cat(y_true_all).numpy()
        y_pred_np = torch.cat(y_pred_all).numpy()
        try:
            val_r2 = r2_score(y_true_np, y_pred_np)
        except Exception:
            val_r2 = float('nan')
    else:
        val_r2 = float('nan')

    return val_loss, val_mae, val_mse, val_r2


def load_pretrained(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')['model_state_dict']
    matched = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.load_state_dict(matched, strict=False)
    print('Pretrained Checkpoint Loaded')
