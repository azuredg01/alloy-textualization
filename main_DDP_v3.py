import os
import sys
import csv
import yaml
import shutil
import warnings
import logging
from datetime import datetime
import time
import pandas as _pd

import torch
import torch.distributed as dist


from accelerate import Accelerator
from transformers.utils import logging as hf_logging

from data.dataloader import load_data
from model.network_v2 import create_model, cri_opt_sch
from model.utils_v2 import train_pt, validate_pt, train_ft, validate_ft, load_pretrained

# =======================================
# [ADD] 進階 CSV helpers & 驗證推論工具
# =======================================
def _init_csv_ex(path: str, header: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()

def _append_csv_ex(path: str, row: dict, header: list):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow({k: row.get(k, None) for k in header})

@torch.no_grad()
def _predict_on_loader(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []

    for batch in data_loader:
        # 安全取得 ids
        ids = batch.get('ids', None)
        if ids is None:
            ids = batch.get('input_ids')
        # 安全取得 attention mask（可能沒有）
        attn = batch.get('mask', None)
        if attn is None:
            attn = batch.get('attention_mask', None)

        ids = ids.to(device)
        attn = attn.to(device) if attn is not None else None

        if attn is not None:
            out = model(input_ids=ids, attention_mask=attn)
        else:
            out = model(input_ids=ids)

        logits = getattr(out, "logits", out)
        pred = logits.squeeze(-1)

        # 收集預測（normalized）
        y_pred.extend(pred.detach().cpu().tolist())

        # 若 batch 有 target（val loader 有），也一起收集
        if isinstance(batch, dict) and ('target' in batch):
            y_true.extend(batch['target'].detach().cpu().tolist())

    return y_true, y_pred

# ======================
# Pretrain
# ======================
def run_pretrain(acc, config, run_name, save_dir):
    # === build ===
    model = create_model(config)

    train_dl, val_dl = load_data(config)
    criterion, optimizer, scheduler = cri_opt_sch(config, model)

    # === prepare ===
    model, optimizer, train_dl, val_dl = acc.prepare(model, optimizer, train_dl, val_dl)

    # ...existing code...

    best_loss = float("inf")
    for epoch in range(config['epochs']):
        train_loss, lr = train_pt(model, train_dl, optimizer, criterion, scheduler, acc.device, acc)
        val_loss = validate_pt(model, val_dl, criterion, acc.device, acc)

        if acc.is_main_process:
            print(f'Epoch {epoch+1}/{config["epochs"]} - '
                  f'Train {train_loss:.6f} | ValLoss {val_loss:.6f} | LR {lr:.2e}')

        if acc.is_main_process and (val_loss <= best_loss):
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': lr
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')

    # ...existing code...


# ======================
# Fine-tune（已加入 3 份 CSV + 時間）
# ======================
def run_finetune(acc, config, run_name, save_dir, time_now):
    # === build ===
    model = create_model(config)

    # if config.get('load_pretrained', False):
    #     load_pretrained(model, config['paths']['pretrained'])

    # # === [ADD] 讀取 train/val pkl，取得 target_raw 與 tmax（反正規化用）===
    # df_train_raw = _pd.read_pickle(config['paths']['train_data'])
    # df_val_raw   = _pd.read_pickle(config['paths']['val_data'])

    # if "target_raw" in df_train_raw.columns:
    #     tmax = df_train_raw["target_raw"].max()
    # else:
    #     raise ValueError("train_data pkl 中沒有 'target_raw' 欄位，無法做反正規化")


    train_dl, val_dl = load_data(config)
    criterion, optimizer, scheduler = cri_opt_sch(config, model)

    # === prepare ===
    model, optimizer, train_dl, val_dl = acc.prepare(model, optimizer, train_dl, val_dl)

    # === [ADD] 進階 CSV 與計時器 ===
    train_log_csv    = os.path.join(save_dir, f"train_log_{time_now}.csv")
    best_summary_csv = os.path.join(save_dir, f"best_model_summary_{time_now}.csv")
    # best_pred_csv    = os.path.join(save_dir, f"best_predictions_{time_now}.csv")
    train_time_csv   = os.path.join(save_dir, f"train_time_{time_now}.csv")

    train_header = [
        "timestamp","run_name","epoch",
        "train_loss","val_loss","val_MAE","val_MSE","val_R2",
        "learning_rate","epoch_seconds"
    ]
    best_header = [
        "timestamp","run_name","best_metric_name","best_metric_value","best_epoch",
        "val_loss","val_MAE","val_MSE","val_R2",
        "model_path","save_dir",
        # "config_json"
    ]
    train_time_header = [
        "timestamp","run_name","start_time","end_time","total_seconds","total_hms",
        # "epochs","best_epoch","best_metric_name","best_metric_value","save_dir","model_path"
    ]

    _init_csv_ex(train_log_csv, train_header)
    _init_csv_ex(best_summary_csv, best_header)
    _init_csv_ex(train_time_csv, train_time_header)

    # 總訓練計時
    run_start_time = time.time()
    run_start_iso  = datetime.now().isoformat(timespec="seconds")

    # 最佳值追蹤（沿用你原本以 val_loss 判斷最佳）
    best_loss = float("inf")
    best_epoch = -1
    best_metric_name = "val_loss"
    best_metric_value = float("inf")

    # ===== [ADD] Early Stopping 設定（由 config.yaml 控制） =====
    es_cfg       = config.get("early_stopping", {}) or {}
    es_enabled   = bool(es_cfg.get("enabled", False))
    es_patience  = int(es_cfg.get("patience", 10))
    es_monitor   = es_cfg.get("monitor", "val_loss")   # 可選: val_loss / val_MAE / val_MSE / val_R2
    es_mode      = es_cfg.get("mode", "min")           # min (loss/MAE/MSE) 或 max (R2)
    es_min_delta = float(es_cfg.get("min_delta", 0.0)) # 最小改善幅度
    es_warmup    = int(es_cfg.get("warmup", 0))        # 前幾個 epoch 不啟動早停
    # Early Stopping 內部狀態
    es_best_value = -float("inf") if es_mode == "max" else float("inf")
    es_no_improve = 0

    # ...existing code...

    # === train ===
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()

        # ===== Epoch-mod warmup（只在 ReduceLROnPlateau 時啟用）=====
        sch_cfg = (config.get('sch') or {})
        warmup_epochs_cfg = sch_cfg.get('warmup_epochs', 0)
        warmup_epoch_ratio = sch_cfg.get('warmup_epoch_ratio', None)

        # 先決定 warmup_epochs：若有給 ratio，用 ratio * 總 epoch，至少 1
        if warmup_epoch_ratio is not None:
            try:
                ratio = float(warmup_epoch_ratio)
            except Exception:
                ratio = 0.0
            warmup_epochs = max(1, int(round(ratio * int(config['epochs'])))) if ratio > 0 else 0
        else:
            warmup_epochs = int(warmup_epochs_cfg or 0)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and warmup_epochs > 0:
            if epoch < warmup_epochs:
                # 線性 warmup（從 0 → base_lr）
                warm_step_ratio = float(epoch + 1) / float(warmup_epochs)
                base_lr = float(config['optim']['lr'])
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr * warm_step_ratio
            else:
                # 過了 warmup，保證不高於 base_lr（Plateau 會往下調）
                base_lr = float(config['optim']['lr'])
                for pg in optimizer.param_groups:
                    pg['lr'] = min(pg['lr'], base_lr)
        # ============================================================

        train_loss, lr = train_ft(model, train_dl, optimizer, criterion, scheduler, acc.device, acc)
        val_loss, val_mae, val_mse, val_r2 = validate_ft(model, val_dl, criterion, acc.device, acc)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch_monitor = (config.get('sch', {}) or {}).get('monitor', 'val_loss')  # val_loss / val_MAE / val_MSE / val_R2
            cur = {
                'val_loss': float(val_loss),
                'val_MAE':  float(val_mae),
                'val_MSE':  float(val_mse),
                'val_R2':   float(val_r2),
            }.get(sch_monitor, float(val_loss))
            scheduler.step(cur)

        if acc.is_main_process:
            print(f'Epoch {epoch+1}/{config["epochs"]} - '
                  f'Train {train_loss:.6f} | ValLoss {val_loss:.6f} | '
                  f'MAE {val_mae:.6f} | MSE {val_mse:.6f} | R2 {val_r2:.6f} | LR {lr:.2e}')

        # 寫入逐 epoch 訓練紀錄
        if acc.is_main_process:
            try:
                lr_now = optimizer.param_groups[0]["lr"]
            except Exception:
                lr_now = None
            epoch_time = time.time() - epoch_start_time
            _append_csv_ex(
                train_log_csv,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "epoch": epoch+1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_MAE": float(val_mae),
                    "val_MSE": float(val_mse),
                    "val_R2":  float(val_r2),
                    "learning_rate": lr_now,
                    "epoch_seconds": round(float(epoch_time), 2),
                },
                train_header
            )

        # ===== Early Stopping 判斷 =====
        if es_enabled:
            cur_metrics = {
                "val_loss": float(val_loss),
                "val_MAE":  float(val_mae),
                "val_MSE":  float(val_mse),
                "val_R2":   float(val_r2),
            }
            cur_val = cur_metrics.get(es_monitor, float(val_loss))

            if (epoch + 1) <= es_warmup:
                # warmup 階段：只更新基準值，不計入 no_improve
                if (es_mode == "min" and cur_val < es_best_value) or (es_mode == "max" and cur_val > es_best_value):
                    es_best_value = cur_val
                es_no_improve = 0
            else:
                improved = ((es_best_value - cur_val) > es_min_delta) if es_mode == "min" \
                           else ((cur_val - es_best_value) > es_min_delta)
                if improved:
                    es_best_value = cur_val
                    es_no_improve = 0
                else:
                    es_no_improve += 1

            stop_now = es_no_improve >= es_patience

            # 多 GPU 同步停止訊號（rank0 決定 → 廣播）
            stop_tensor = torch.tensor([1 if stop_now else 0], device=acc.device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(stop_tensor, src=0)

            if int(stop_tensor.item()) == 1:
                if acc.is_main_process:
                    print(f"[EARLY-STOP] monitor={es_monitor} mode={es_mode} "
                          f"no_improve={es_no_improve} at epoch {epoch+1}")
                break
        # ===============================

        # === 刷新最佳就存檔 + 輸出最佳預測 + 寫入摘要 ===
        if acc.is_main_process and (val_loss <= best_loss) and not config['debug']:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_mse': val_mse,
                'val_r2': val_r2,
                'lr': lr
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')

            # # [ADD] 產生最佳模型在驗證集逐樣本預測（含反正規化，無誤差）
            # try:
            #     # 模型輸出的 normalized y_true / y_pred
            #     y_true_norm, y_pred_norm = _predict_on_loader(model, val_dl, acc.device)
            #     if isinstance(y_true_norm, torch.Tensor):
            #         y_true_norm = y_true_norm.detach().cpu().tolist()
            #     if isinstance(y_pred_norm, torch.Tensor):
            #         y_pred_norm = y_pred_norm.detach().cpu().tolist()

            #     y_true_raw = [y * tmax for y in y_true_norm]
            #     y_pred_raw = [p * tmax for p in y_pred_norm]

            #     # （1）寫 normalized 版本（覆蓋原 best_predictions_xxx.csv）
            #     df_pred_norm = _pd.DataFrame({
            #         "index": list(range(len(y_pred_norm))),
            #         "y_true_norm": y_true_norm if len(y_true_norm) == len(y_pred_norm) else [None]*len(y_pred_norm),
            #         "y_pred_norm": y_pred_norm,
            #     })
            #     df_pred_norm.to_csv(best_pred_csv, index=False)

            #     # （2）寫未正規化版本（新增一份
            #     print("len(y_true_raw):", len(y_true_raw))
            #     print("len(y_pred_raw):", len(y_pred_raw))

            #     df_pred_denorm = _pd.DataFrame({
            #         "index": list(range(len(y_pred_norm))),
            #         "y_true_raw": y_true_raw,
            #         "y_pred_raw": y_pred_raw,
            #     })

            #     denorm_csv = os.path.join(save_dir, f"best_predictions_denorm_{time_now}.csv")
            #     df_pred_denorm.to_csv(denorm_csv, index=False)
            #     print(f"[INFO] 已寫入反正規化預測：{denorm_csv}")

            # except Exception as _e:
            #     print(f"[WARN] failed to write best_predictions.csv: {_e}")

            # [ADD] 寫入最佳摘要一列（可追蹤每次被刷新）
            best_epoch = epoch+1
            best_metric_value = float(val_loss)
            _append_csv_ex(
                best_summary_csv,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_metric_value,
                    "best_epoch": best_epoch,
                    "val_loss": float(val_loss),
                    "val_MAE": float(val_mae),
                    "val_MSE": float(val_mse),
                    "val_R2":  float(val_r2),
                    "model_path": f"{save_dir}/model.pt",
                    "save_dir": save_dir,
                    # "config_json": yaml.safe_dump(config, allow_unicode=True)
                },
                best_header
            )

    # === [ADD] 訓練結束後：寫入總訓練時間 ===
    end_iso = datetime.now().isoformat(timespec="seconds")
    total_sec = time.time() - run_start_time
    def _sec_to_hms(sec: float) -> str:
        s = int(round(sec))
        h = s // 3600
        m = (s % 3600) // 60
        s = s % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    total_hms = _sec_to_hms(total_sec)
    if acc.is_main_process:
        _append_csv_ex(
            train_time_csv,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_name": run_name,
                "start_time": run_start_iso,
                "end_time": end_iso,
                "total_seconds": round(float(total_sec), 3),
                "total_hms": total_hms,
                # "epochs": int(config.get("epochs", 0)),
                # "best_epoch": best_epoch if best_epoch != -1 else None,
                # "best_metric_name": best_metric_name,
                # "best_metric_value": best_metric_value if best_epoch != -1 else None,
                # "save_dir": save_dir,
                # "model_path": f"{save_dir}/model.pt",
            },
            train_time_header
        )

    if acc.is_main_process:
        # 補打一遍 checkpoint 摘要
        ckpt = torch.load(f'{save_dir}/model.pt', map_location='cpu')
        print(f"Best val MSE(loss-batchavg): {ckpt['val_loss']:.6f}")
        if 'val_mse' in ckpt: print(f"Best val MSE: {ckpt['val_mse']:.6f}")
        if 'val_mae' in ckpt: print(f"Best val MAE: {ckpt['val_mae']:.6f}")
        if 'val_r2'  in ckpt: print(f"Best val R2 : {ckpt['val_r2']:.6f}")

# ======================
# Main
# ======================
def main():
    # ========== 0) 建 Accelerator，壓低子進程輸出 ==========
    acc = Accelerator()
    if not acc.is_main_process:
        hf_logging.set_verbosity_error()
        logging.getLogger().setLevel(logging.ERROR)
        sys.excepthook = lambda *args, **kwargs: None
        warnings.filterwarnings("ignore")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # ========== 1) 載 config ==========
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-tcf', '--training_config', type=str, default='config/config.yaml', help='Training Config yaml 路徑')
    args = parser.parse_args()


    config_path = args.training_config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    config['device'] = str(acc.device)
    torch.backends.cudnn.benchmark = True

    # ========== 2) 設定路徑/檔名；主進程做 I/O ==========
    time_now = datetime.now().strftime("%m%d_%H%M%S")
    run_name  = f'{config["stage"]}-{time_now}'
    save_dir  = f'./training_output/{config["stage"]}/{run_name}'

    if acc.is_main_process:
        print(f'Device: {acc.device}\n')
        os.makedirs(save_dir, exist_ok=True)
        # 可選：把設定與網路結構複製到輸出資料夾做紀錄
        try:
            shutil.copy(config_path, f'{save_dir}/config.yaml')
        except Exception as e:
            print(f'[WARN] copy config.yaml failed: {e}')
        try:
            shutil.copy('./model/network_v2.py', f'{save_dir}/network_v2.py')
        except Exception as e:
            print(f'[WARN] copy network_v2.py failed: {e}')

        # ...existing code...

    # 所有進程都印出啟動訊息（用 acc.print 聚合）
    acc.print(f'=== TRAINING ({config["stage"]}) | rank {acc.process_index} on {acc.device} ===')

    # ========== 3) 依 stage 跑 ==========
    if config["stage"] == 'pretrain':
        run_pretrain(acc, config, run_name, save_dir)
    else:
        run_finetune(acc, config, run_name, save_dir, time_now)

    # ========== 4) 正常收尾 ==========
    acc.wait_for_everyone()



if __name__ == "__main__":
    main()
