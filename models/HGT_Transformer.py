import math
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm


SEQ_LEN = 3
HIDDEN_DIM = 128
TRANSFORMER_DIM = 128
NHEAD = 4
NLAYERS = 2
LR = 1e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "trained_hgt_transformer.pt"
RESULTS_DIR = "D:/HGT-Transformer/results"
HGT_LAYERS = 2
HGT_HEADS = 4

os.makedirs(RESULTS_DIR, exist_ok=True)



def build_window_sample(multi_dataset, start_idx, seq_len=SEQ_LEN):
    if start_idx < 0 or start_idx + seq_len - 1 >= len(multi_dataset):
        raise IndexError(f"build_window_sample: start_idx {start_idx} 超出范围 len={len(multi_dataset)}")

    day0 = multi_dataset[start_idx]
    node_types = list(day0.node_types)
    type_x, type_pos, rel_edge_lists = {}, {}, {}
    time_feats, time_pos = [], []

    for k in range(seq_len):
        day = multi_dataset[start_idx + k]
        for nt in day.node_types:
            type_x.setdefault(nt, [])
            type_pos.setdefault(nt, [])
            x = day[nt].x if "x" in day[nt] else torch.zeros((0, 1))
            pos = day[nt].pos if "pos" in day[nt] else None
            type_x[nt].append(x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x))
            if pos is not None:
                type_pos[nt].append(pos.detach().cpu())

        for et in day.edge_types:
            try:
                e = day[et].edge_index if et in day else None
            except Exception:
                e = None
            if e is None or e.numel() == 0:
                continue
            rel_edge_lists.setdefault(et, []).append((k, e.detach().cpu().long()))

        if "time" in day.node_types:
            time_feats.append(day["time"].x.detach().cpu())
            time_pos.append(day["time"].pos.detach().cpu() if "pos" in day["time"] else torch.zeros((day["time"].x.size(0), 2)))
        else:
            time_feats.append(torch.zeros((1, 1)))
            time_pos.append(torch.zeros((1, 2)))

    window = HeteroData()
    offsets = {}

    for nt, blocks in type_x.items():
        if len(blocks) == 0:
            window[nt].x = torch.zeros((0, 1))
            window[nt].pos = torch.zeros((0, 2))
            offsets[nt] = [0] * seq_len
            continue
        xs = torch.cat(blocks, dim=0)
        window[nt].x = xs.float()
        if nt in type_pos and len(type_pos[nt]) > 0:
            window[nt].pos = torch.cat(type_pos[nt], dim=0).float()
        else:
            window[nt].pos = torch.zeros((xs.size(0), 2))
        counts = [b.shape[0] for b in blocks]
        offs, acc = [], 0
        for c in counts:
            offs.append(acc)
            acc += int(c)
        offsets[nt] = offs + [acc] * (seq_len - len(offs))

    # === 合并边 ===
    for et, lst in rel_edge_lists.items():
        src_type, rel, dst_type = et
        edge_tensors = []
        for (k, e_cpu) in lst:
            if src_type not in offsets or dst_type not in offsets:
                continue
            src_off = int(offsets[src_type][k])
            dst_off = int(offsets[dst_type][k])
            e_off = e_cpu.clone()
            e_off[0, :] += src_off
            e_off[1, :] += dst_off
            edge_tensors.append(e_off)
        window[et].edge_index = torch.cat(edge_tensors, dim=1) if len(edge_tensors) > 0 else torch.zeros((2, 0), dtype=torch.long)

    try:
        time_x = torch.cat(time_feats, dim=0)
        time_pos = torch.cat(time_pos, dim=0)
    except Exception:
        time_x = torch.zeros((seq_len, 1))
        time_pos = torch.zeros((seq_len, 2))
    window["time"].x = time_x.float()
    window["time"].pos = time_pos.float()

    Nt = time_x.size(0)
    if Nt > 1:
        edges = torch.tensor([[i, i + 1] for i in range(Nt - 1)] + [[i + 1, i] for i in range(Nt - 1)], dtype=torch.long).T
        window["time", "temporal", "time"].edge_index = edges
    else:
        window["time", "temporal", "time"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    for nt in window.node_types:
        if "x" in window[nt] and window[nt].x.numel() > 0:
            x = window[nt].x
            mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True)
            std[std < 1e-6] = 1.0
            window[nt].x = ((x - mean) / std).float()

    return window

class HGTSpatialEncoder(nn.Module):
    def __init__(self, in_dims, metadata, hidden_dim=HIDDEN_DIM, heads=HGT_HEADS, num_layers=HGT_LAYERS):
        super().__init__()

        # ★ 正确解包 metadata (PyG: metadata = (node_types, edge_types))
        node_types, edge_types = metadata

        # 统一投影层（不同节点类型 → hidden_dim）
        self.node_proj = nn.ModuleDict()
        for nt in node_types:
            self.node_proj[nt] = nn.Linear(in_dims[nt], hidden_dim)

        self.hidden_dim = hidden_dim

        # ★ HGTConv 层
        self.layers = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(node_types, edge_types),
                heads=heads
            )
            for _ in range(num_layers)
        ])

        # ★ HGT 输出维度的 head 合并
        conv_out_dim = (hidden_dim // heads) * heads
        if conv_out_dim == hidden_dim:
            self.post_proj = nn.Identity()
        else:
            self.post_proj = nn.Linear(conv_out_dim, hidden_dim)

        self.act = nn.ReLU()

    def forward(self, x_dict, edge_index_dict):
        out = {}
        for nt, x in x_dict.items():
            if nt not in self.node_proj:
                raise KeyError(
                    f"HGTSpatialEncoder 未找到节点类型的投影层: '{nt}'. 可用 keys = {list(self.node_proj.keys())}")
            out[nt] = self.node_proj[nt](x)

        for nt, x in out.items():
            if x.size(-1) != self.hidden_dim:
                raise RuntimeError(f"[维度错误] 节点 {nt} 投影后最后一维 = {x.size(-1)}, 但期望 = {self.hidden_dim}")

        # --- HGT 传播 ---
        for conv in self.layers:
            h = conv(out, edge_index_dict)
            out = {k: self.act(self.post_proj(v)) for k, v in h.items()}

        return out


class TimeTransformer(nn.Module):
    def __init__(self, dim=TRANSFORMER_DIM, nhead=NHEAD, nlayers=NLAYERS):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                                   dim_feedforward=dim * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.input_proj = nn.Linear(HIDDEN_DIM, dim)
        self.out_head = nn.Linear(dim, 1)

    def forward(self, seq):
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        x = self.input_proj(seq)
        out = self.transformer(x)
        return self.out_head(out).squeeze(-1).squeeze(0)


class HGTTransformerModel(nn.Module):
    def __init__(self, node_input_dims, metadata, hidden_dim=HIDDEN_DIM, time_dim=TRANSFORMER_DIM):
        super().__init__()
        self.spatial = HGTSpatialEncoder(node_input_dims, metadata, hidden_dim)
        self.temporal = TimeTransformer(dim=time_dim)
        self.decoder = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, data):
        device = data['time'].x.device
        x_dict = {nt: (data[nt].x if "x" in data[nt] else torch.zeros((0, 1))).to(device) for nt in data.node_types}
        edge_index_dict = {et: data[et].edge_index.to(device).long() if hasattr(data[et], "edge_index")
                           else torch.zeros((2, 0), dtype=torch.long, device=device)
                           for et in data.edge_types}

        enc = self.spatial(x_dict, edge_index_dict)
        time_num = data['time'].x.size(0)
        if len(enc) == 0 or time_num == 0:
            return torch.zeros((time_num,), device=device)

        hidden = next(iter(enc.values())).size(1)
        time_emb = torch.zeros((time_num, hidden), device=device)
        total_c = torch.zeros((time_num,), device=device)

        for src in enc.keys():
            if src == 'time': continue
            key = (src, 'belong_to', 'time')
            if key not in edge_index_dict: continue
            e = edge_index_dict[key]
            src_idx, dst_idx = e[0], e[1]
            valid = (dst_idx < time_num) & (src_idx < enc[src].size(0))
            src_idx, dst_idx = src_idx[valid], dst_idx[valid]
            time_emb.index_add_(0, dst_idx, enc[src][src_idx])
            total_c.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=torch.float))

        total_c[total_c == 0] = 1.0
        time_emb = time_emb / total_c.unsqueeze(-1)
        preds = self.temporal(time_emb)
        preds = self.decoder(preds.unsqueeze(-1)).squeeze(-1)
        return preds

def train_epoch(model, optimizer, dataset, start_indices, device, target_scaler):
    model.train()
    criterion = nn.MSELoss()

    t_mean, t_std = target_scaler if target_scaler is not None else (0.0, 1.0)

    per_window = []
    all_preds = []
    all_trues = []
    losses = []

    pbar = tqdm(start_indices, desc="训练窗口进度", unit="窗口", leave=False)

    for start in pbar:
        try:
            data = build_window_sample(dataset, start, seq_len=SEQ_LEN)
        except Exception as e:
            warnings.warn(f"build_window_sample failed for start {start}: {e}. 跳过该窗口。")
            continue

        data = data.to(device)

        # forward
        preds = model(data)
        if preds.numel() == 0:
            warnings.warn(f"模型输出为空 preds.numel()==0 at start {start}. 跳过样本。")
            continue

        y_pred_torch = preds[-1].unsqueeze(0)
        y_pred_scalar = float(y_pred_torch.detach().cpu().item())

        true_val_raw = None
        try:
            cams_ds = getattr(dataset, "cams", None)
            if cams_ds is not None:
                cams_day_index = start + SEQ_LEN - 1
                if 0 <= cams_day_index < len(cams_ds):
                    cams_day = cams_ds[cams_day_index]

                    if "cams_grid" in cams_day.node_types:
                        x = cams_day["cams_grid"].x
                        if x.numel() > 0:
                            # ★ CO₂ 真实值 = CAMS 网格点 CO₂ 均值
                            true_val_raw = float(x.mean().item())

        except Exception as e:
            warnings.warn(f"提取 CAMS CO₂ 真值异常: {e}")

        if true_val_raw is None:
            try:
                if "cams_grid" in data.node_types:
                    x = data["cams_grid"].x
                    if x.numel() > 0:
                        true_val_raw = float(x.mean().item())
            except:
                pass

        if true_val_raw is None:
            true_val_raw = 0.0
            warnings.warn(f"真实值无法提取 at start={start}。")

        if target_scaler is not None and t_std != 0:
            true_val_norm = (true_val_raw - t_mean) / t_std
        else:
            true_val_norm = true_val_raw

        y_true_torch = torch.tensor([true_val_norm], device=device, dtype=torch.float32)

        loss = criterion(y_pred_torch, y_true_torch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # window metrics
        err = (y_pred_scalar - float(true_val_norm))
        mse_win = err * err
        rmse_win = float(np.sqrt(mse_win))
        mae_win = float(abs(err))

        losses.append(float(loss.item()))
        all_preds.append(y_pred_scalar)
        all_trues.append(float(true_val_norm))

        per_window.append({
            "start": int(start),
            "loss": float(loss.item()),
            "rmse": rmse_win,
            "mae": mae_win,
            "true_norm": float(true_val_norm),
            "pred_norm": float(y_pred_scalar),
            "true_raw": float(true_val_raw),
            "pred_raw": float(y_pred_scalar * t_std + t_mean)
                         if (target_scaler is not None and t_std != 0)
                         else float(y_pred_scalar)
        })

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "RMSE": f"{rmse_win:.4f}",
            "MAE": f"{mae_win:.4f}"
        })

    # ---- epoch metrics ----
    if len(all_preds) == 0:
        epoch_mse = epoch_rmse = epoch_mae = epoch_r2 = float("nan")
    else:
        epoch_mse = mean_squared_error(all_trues, all_preds)
        epoch_rmse = float(np.sqrt(epoch_mse))
        epoch_mae = mean_absolute_error(all_trues, all_preds)
        epoch_r2 = r2_score(all_trues, all_preds) if len(all_preds) > 1 else float("nan")

    print(f"\n📊 本 Epoch 训练整体指标 (标准化域): MSE={epoch_mse:.6f}, RMSE={epoch_rmse:.6f}, MAE={epoch_mae:.6f}, R²={epoch_r2 if not np.isnan(epoch_r2) else 'NaN'}")

    return {
        "loss": float(np.mean(losses)) if len(losses) > 0 else float("nan"),
        "mse": float(epoch_mse),
        "rmse": float(epoch_rmse),
        "mae": float(epoch_mae),
        "r2": float(epoch_r2) if not np.isnan(epoch_r2) else None,
        "per_window": per_window
    }



def eval_model(model, dataset, start_indices, device, target_scaler=None):
    model.eval()
    preds_all = []
    trues_all = []

    with torch.no_grad():
        for start in tqdm(start_indices, desc="评估窗口进度", unit="窗口", leave=False):
            try:
                data = build_window_sample(dataset, start, seq_len=SEQ_LEN)
            except:
                continue
            data = data.to(device)
            preds = model(data)
            if preds.numel() == 0:
                continue
            y_pred_norm = float(preds[-1].cpu().item())

            # 真实值取自 cams grid.mean（raw）
            true_val_raw = None
            try:
                cams_last = getattr(dataset, "cams", None)
                if cams_last is not None:
                    cams_day = cams_last[start + SEQ_LEN - 1]
                    if "cams_grid" in cams_day.node_types and cams_day["cams_grid"].x.numel() > 0:
                        true_val_raw = float(cams_day["cams_grid"].x.mean().cpu().item())
            except Exception:
                pass

            if true_val_raw is None:
                # fallback to window time last
                try:
                    if "cams_grid" in data.node_types and data["cams_grid"].x.numel() > 0:
                        true_val_raw = float(data["cams_grid"].x.mean().cpu().item())
                except Exception:
                    continue

            # 若标准化过，需要反归一化预测
            if target_scaler is not None:
                t_mean, t_std = target_scaler
                y_pred = y_pred_norm * t_std + t_mean
            else:
                y_pred = y_pred_norm

            preds_all.append(y_pred)
            trues_all.append(true_val_raw)

    if len(preds_all) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), [], []

    preds_all = np.array(preds_all)
    trues_all = np.array(trues_all)

    mse = np.mean((preds_all - trues_all) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_all - trues_all))
    r2 = r2_score(trues_all, preds_all) if len(preds_all) > 1 else float("nan")

    print(f"测试集（反归一化）: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    return mse, rmse, mae, r2, trues_all.tolist(), preds_all.tolist()


def compute_target_stats(dataset, train_idx):
    vals = []
    for start in train_idx:
        try:
            cams_last = dataset.cams[start + SEQ_LEN - 1]
            if "cams_grid" in cams_last.node_types and cams_last["cams_grid"].x.numel() > 0:
                vals.extend(cams_last["cams_grid"].x.detach().cpu().numpy().ravel())
        except Exception:
            continue
    vals = np.array(vals)
    if vals.size == 0:
        return 0.0, 1.0
    return float(vals.mean()), float(vals.std() + 1e-6)


def main():
    from graphs.era5 import ERA5Dataset
    from graphs.oco2 import OCO2Dataset
    from graphs.cams import CAMSDataset
    from graphs.data import MultiSourceDataset

    print("正在加载数据集...")
    era = ERA5Dataset("D:/HGT-Transformer/data/ERA-5")
    oco = OCO2Dataset("D:/HGT-Transformer/data/OCO-2")
    cams = CAMSDataset("D:/HGT-Transformer/processed_data/CAMS-IO-interpolation.nc")
    multi = MultiSourceDataset(datas=[era, oco, cams])
    multi.cams = cams

    total_days = len(multi)
    start_cnt = total_days - (SEQ_LEN - 1)
    train_idx = list(range(int(start_cnt * 0.8)))
    test_idx = list(range(int(start_cnt * 0.8), start_cnt))

    y_mean, y_std = compute_target_stats(multi, train_idx)
    target_scaler = (y_mean, y_std)
    print(f"XCO₂归一化: mean={y_mean:.3f}, std={y_std:.3f}")

    sample = build_window_sample(multi, 0)
    node_input_dims = {nt: sample[nt].x.size(1) for nt in sample.node_types}
    metadata = (list(sample.node_types), list(sample.edge_types))
    model = HGTTransformerModel(node_input_dims, metadata).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    train_losses, test_rmse = [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        loss = train_epoch(model, optimizer, multi, train_idx, DEVICE, target_scaler)
        mse, rmse, mae, r2, trues, preds = eval_model(model, multi, test_idx, DEVICE, target_scaler)
        train_losses.append(loss)
        test_rmse.append(rmse)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型保存至 {MODEL_SAVE_PATH}")

    # 提取标量序列
    train_loss_curve = [item["loss"] for item in train_losses]
    test_rmse_curve = test_rmse  # 这个本来就是 list[float]

    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_curve, label="Train Loss")
    plt.plot(test_rmse_curve, label="Test RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "train_curve.png"), dpi=300)

    print("曲线保存完毕。")


if __name__ == "__main__":
    main()
