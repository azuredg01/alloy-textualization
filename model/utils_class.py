# utils_DDP.py —— Accelerate-safe, Classification-only (Acc / Sens / Spec / AUC)
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize


# ---------------- helpers ----------------
def _get_attn(batch):
    """Robustly fetch attention_mask without tensor boolean ops."""
    attn = batch.get("mask", None) if isinstance(batch, dict) else None
    if attn is None and isinstance(batch, dict):
        attn = batch.get("attention_mask", None)
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


# ======================
# Pretrain (MLM)
# ======================
def train_pt(model, dataloader, optimizer, scheduler, device, acc=None):
    model.train()
    total_loss, lr_sum = 0.0, 0.0

    for batch in _tqdm(dataloader, acc):
        ids = _to(batch["ids"], device)
        attn = _to(batch["mask"], device) if "mask" in batch else _to(
            batch["attention_mask"], device
        )
        labels = _to(batch["labels"], device)

        optimizer.zero_grad(set_to_none=True)

        out = model(ids, attention_mask=attn, labels=labels)
        loss = out.loss

        if acc is not None:
            acc.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        lr_sum += optimizer.param_groups[0]["lr"]

    return (total_loss / max(1, len(dataloader))), (lr_sum / max(1, len(dataloader)))


@torch.no_grad()
def validate_pt(model, dataloader, device, acc=None):
    model.eval()
    total_loss = 0.0

    for batch in _tqdm(dataloader, acc):
        ids = _to(batch["ids"], device)
        attn = _to(batch["mask"], device) if "mask" in batch else _to(
            batch["attention_mask"], device
        )
        labels = _to(batch["labels"], device)

        out = model(ids, attention_mask=attn, labels=labels)
        loss = out.loss
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


# ======================
# Fine-tune（Classification only）
# ======================
def train_ft(model, dataloader, optimizer, criterion, scheduler, device, acc=None):
    """
    Fine-tune：分類專用
    - logits: (B, num_labels)
    - target: (B,) long
    """
    model.train()
    total_loss, lr_sum = 0.0, 0.0

    for batch in _tqdm(dataloader, acc):
        if isinstance(batch, dict):
            ids = _to(batch["ids"], device)
            attn = _get_attn(batch)
            attn = _to(attn, device) if attn is not None else None
            target = _to(batch["target"], device)
        else:
            if len(batch) == 3:
                ids, attn, target = batch
                ids = _to(ids, device)
                attn = _to(attn, device) if attn is not None else None
                target = _to(target, device)
            else:
                ids, target = batch
                ids = _to(ids, device)
                target = _to(target, device)
                attn = None

        optimizer.zero_grad(set_to_none=True)

        logits = _forward_with_mask(model, ids, attn)  # (B, C)
        loss = criterion(logits, target)               # CrossEntropyLoss

        if acc is not None:
            acc.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        # step-based scheduler 在 batch 內 step；Plateau 由 main 在 epoch 後 step(...)
        if (
            scheduler is not None
            and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            )
        ):
            scheduler.step()

        total_loss += loss.item()
        lr_sum += optimizer.param_groups[0]["lr"]

    return (total_loss / max(1, len(dataloader))), (lr_sum / max(1, len(dataloader)))


@torch.no_grad()
def validate_ft(model, val_loader, criterion, device, acc=None):
    """
    分類評估：
        - val_loss: CrossEntropyLoss
        - val_acc : Accuracy
        - val_sens: Sensitivity (macro recall)
        - val_spec: Specificity (macro specificity)
        - val_auc : AUC (binary / multi-class OVR)
    """
    model.eval()

    total_loss = 0.0
    n_batches = 0

    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for batch in _tqdm(val_loader, acc):
        if isinstance(batch, dict):
            ids = _to(batch["ids"], device)
            attn = _get_attn(batch)
            attn = _to(attn, device) if attn is not None else None
            target = _to(batch["target"], device)
        else:
            if len(batch) == 3:
                ids, attn, target = batch
                ids = _to(ids, device)
                attn = _to(attn, device) if attn is not None else None
                target = _to(target, device)
            else:
                ids, target = batch
                ids = _to(ids, device)
                target = _to(target, device)
                attn = None

        logits = _forward_with_mask(model, ids, attn)  # (B, C)
        loss = criterion(logits, target)

        proba = torch.softmax(logits, dim=-1)
        pred_labels = logits.argmax(dim=-1)

        if acc is not None:
            target_g = acc.gather_for_metrics(target)
            pred_g = acc.gather_for_metrics(pred_labels)
            proba_g = acc.gather_for_metrics(proba)
        else:
            target_g, pred_g, proba_g = target, pred_labels, proba

        y_true_all.append(target_g.detach().cpu())
        y_pred_all.append(pred_g.detach().cpu())
        y_proba_all.append(proba_g.detach().cpu())

        total_loss += loss.item()
        n_batches += 1

    val_loss = total_loss / max(1, n_batches)

    if not y_true_all:
        return val_loss, float("nan"), float("nan"), float("nan"), float("nan")

    y_true_np = torch.cat(y_true_all).numpy()
    y_pred_np = torch.cat(y_pred_all).numpy()
    proba_np = torch.cat([p.reshape(-1, p.shape[-1]) for p in y_proba_all]).numpy()

    classes = np.unique(y_true_np)
    accuracy = (y_true_np == y_pred_np).mean().item()

    sensitivity = float("nan")
    specificity = float("nan")
    auc_value = float("nan")

    if len(classes) == 2:
        # Binary classification
        cm = confusion_matrix(y_true_np, y_pred_np, labels=classes)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        pos_cls = classes.max()
        pos_index = int(pos_cls)
        try:
            auc_value = roc_auc_score(
                (y_true_np == pos_cls).astype(int),
                proba_np[:, pos_index],
            )
        except Exception:
            auc_value = float("nan")

    else:
        # Multi-class：macro one-vs-rest
        sens_list, spec_list = [], []
        for cls in classes:
            y_true_bin = (y_true_np == cls).astype(int)
            y_pred_bin = (y_pred_np == cls).astype(int)
            cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                if (tp + fn) > 0:
                    sens_list.append(tp / (tp + fn))
                if (tn + fp) > 0:
                    spec_list.append(tn / (tn + fp))

        if sens_list:
            sensitivity = float(np.mean(sens_list))
        if spec_list:
            specificity = float(np.mean(spec_list))

        try:
            cls_sorted = np.sort(classes)
            y_true_bin_all = label_binarize(y_true_np, classes=cls_sorted)
            auc_value = roc_auc_score(
                y_true_bin_all,
                proba_np[:, cls_sorted],
                average="macro",
                multi_class="ovr",
            )
        except Exception:
            auc_value = float("nan")

    print(
        f"[Val][CLS] Acc={accuracy:.4f}, "
        f"Sens={sensitivity:.4f}, Spec={specificity:.4f}, AUC={auc_value:.4f}"
    )

    val_acc = accuracy
    val_sens = sensitivity
    val_spec = specificity
    val_auc = auc_value

    return val_loss, val_acc, val_sens, val_spec, val_auc


def load_pretrained(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location="cpu")[
        "model_state_dict"
    ]
    matched = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.load_state_dict(matched, strict=False)
    print("Pretrained Checkpoint Loaded")
