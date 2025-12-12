# graphs/data.py
import torch
import numpy as np
from torch_geometric.data import Dataset, HeteroData
from scipy.spatial import cKDTree

RADIUS_DEG_DEFAULT = 0.5
MAX_NEIGHBORS_DEFAULT = 8
TIME_FEATURE_DIM = 4   # 与你的 align_time_feature 保持一致

def build_spatial_links(src_pos, tgt_pos, radius_deg=RADIUS_DEG_DEFAULT, max_neighbors=MAX_NEIGHBORS_DEFAULT):
    if isinstance(src_pos, torch.Tensor):
        src_pos = src_pos.detach().cpu().numpy()
    if isinstance(tgt_pos, torch.Tensor):
        tgt_pos = tgt_pos.detach().cpu().numpy()

    if len(src_pos) == 0 or len(tgt_pos) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    tree = cKDTree(tgt_pos)
    neighbors = tree.query_ball_point(src_pos, r=radius_deg, return_sorted=True)

    rows, cols = [], []
    for i, nbrs in enumerate(neighbors):
        if not nbrs:
            continue
        # 限制邻居数并按距离裁剪
        if max_neighbors and len(nbrs) > max_neighbors:
            d = np.sqrt(((tgt_pos[nbrs] - src_pos[i]) ** 2).sum(axis=1))
            nbrs = np.array(nbrs)[np.argsort(d)[:max_neighbors]]
        for j in nbrs:
            rows.append(i)
            cols.append(int(j))

    if not rows:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)

def align_time_feature(src_name, time_x):
    if not isinstance(time_x, torch.Tensor):
        time_x = torch.tensor(time_x, dtype=torch.float32)

    # 输入 time_x 期望为 (1, n) 或 (m, n) 行数通常为 1（单日）
    aligned = torch.zeros((time_x.shape[0], TIME_FEATURE_DIM), dtype=torch.float32)

    src = src_name.lower()
    if "era" in src:
        n = min(time_x.shape[1], TIME_FEATURE_DIM - 1)
        aligned[:, 1:1 + n] = time_x[:, :n]
    elif "oco" in src or "cam" in src:
        n = min(time_x.shape[1], 1)
        aligned[:, 0] = time_x[:, 0]
    else:
        n = min(time_x.shape[1], TIME_FEATURE_DIM)
        aligned[:, :n] = time_x[:, :n]

    return aligned

class MultiSourceDataset(Dataset):
    def __init__(self, datas, radius_deg=RADIUS_DEG_DEFAULT, max_neighbors=MAX_NEIGHBORS_DEFAULT):
        super().__init__()
        self.sources = datas
        self.radius_deg = radius_deg
        self.max_neighbors = max_neighbors

        self.src_names = []
        for ds in datas:
            name = type(ds).__name__.upper()
            if "ERA" in name:
                self.src_names.append("ERA5")
            elif "OCO" in name:
                self.src_names.append("OCO2")
            elif "CAM" in name:
                self.src_names.append("CAMS")
            else:
                self.src_names.append("SRC")

        # align time length: use min length among sources (all sources expected to be indexed by day)
        self.min_len = min(len(ds) for ds in datas) if len(datas) > 0 else 0

    def __len__(self):
        return self.min_len

    def len(self):
        return self.min_len

    def get(self, idx):

        if idx < 0 or idx >= self.min_len:
            raise IndexError("索引越界")

        hetero = HeteroData()
        src_nodes = {}

        unified_time_list = []

        for src_name, ds in zip(self.src_names, self.sources):
            src_key = src_name.lower()
            day = ds[idx]

            # --- grid ---
            if "grid" in day.node_types:
                tgt = f"{src_key}_grid"
                x_g = day["grid"].x.clone() if hasattr(day["grid"], "x") else torch.zeros((0, 1), dtype=torch.float32)
                hetero[tgt].x = x_g
                if hasattr(day["grid"], "pos"):
                    hetero[tgt].pos = day["grid"].pos.clone()
                else:
                    hetero[tgt].pos = torch.zeros((x_g.size(0), 2), dtype=torch.float32)

                src_nodes.setdefault(src_key, {})
                src_nodes[src_key]["grid_pos"] = hetero[tgt].pos

                if ("grid", "spatial", "grid") in day.edge_types:
                    hetero[tgt, "spatial", tgt].edge_index = day["grid", "spatial", "grid"].edge_index.clone()

                # redirect belong_to -> global time node 0
                if ("grid", "belong_to", "time") in day.edge_types:
                    g2t_local = day["grid", "belong_to", "time"].edge_index.clone()
                    if g2t_local.numel() == 0:
                        hetero[tgt, "belong_to", "time"].edge_index = torch.zeros((2, 0), dtype=torch.long)
                    else:
                        src_idxs = g2t_local[0].clone()
                        dst_idxs = torch.zeros_like(src_idxs)
                        hetero[tgt, "belong_to", "time"].edge_index = torch.stack([src_idxs, dst_idxs], dim=0)

            # --- obs ---
            if "obs" in day.node_types:
                tgt = f"{src_key}_obs"
                x_o = day["obs"].x.clone() if hasattr(day["obs"], "x") else torch.zeros((0, 1), dtype=torch.float32)
                hetero[tgt].x = x_o
                if hasattr(day["obs"], "pos"):
                    hetero[tgt].pos = day["obs"].pos.clone()
                else:
                    hetero[tgt].pos = torch.zeros((x_o.size(0), 2), dtype=torch.float32)

                src_nodes.setdefault(src_key, {})
                src_nodes[src_key]["obs_pos"] = hetero[tgt].pos

                if ("obs", "spatial", "obs") in day.edge_types:
                    hetero[tgt, "spatial", tgt].edge_index = day["obs", "spatial", "obs"].edge_index.clone()

                # redirect obs->time -> global time node 0
                if ("obs", "belong_to", "time") in day.edge_types:
                    o2t_local = day["obs", "belong_to", "time"].edge_index.clone()
                    if o2t_local.numel() == 0:
                        hetero[tgt, "belong_to", "time"].edge_index = torch.zeros((2, 0), dtype=torch.long)
                    else:
                        src_idxs = o2t_local[0].clone()
                        dst_idxs = torch.zeros_like(src_idxs)
                        hetero[tgt, "belong_to", "time"].edge_index = torch.stack([src_idxs, dst_idxs], dim=0)

            # --- time ---
            if "time" in day.node_types:
                raw_time_x = day["time"].x.clone()
                aligned = align_time_feature(src_name, raw_time_x)

                tgt_time = f"{src_key}_time"
                hetero[tgt_time].x = aligned
                # pos: if missing, set zeros with correct rows
                if hasattr(day["time"], "pos"):
                    hetero[tgt_time].pos = day["time"].pos.clone()
                else:
                    hetero[tgt_time].pos = torch.zeros((aligned.size(0), 2), dtype=torch.float32)

                if ("time", "temporal", "time") in day.edge_types:
                    hetero[tgt_time, "temporal", tgt_time].edge_index = \
                        day["time", "temporal", "time"].edge_index.clone()

                unified_time_list.append(aligned)
            else:
                unified_time_list.append(torch.zeros((1, TIME_FEATURE_DIM), dtype=torch.float32))

        # 横向拼接每个 source 的 aligned time -> (1, TIME_FEATURE_DIM * n_sources)
        if len(unified_time_list) > 0:
            # 确保每一项是 (1, TIME_FEATURE_DIM)
            cols = [t if (t.dim() == 2 and t.size(0) == 1) else t.reshape(1, -1) for t in unified_time_list]
            unified_time = torch.cat(cols, dim=1).float()
        else:
            unified_time = torch.zeros((1, TIME_FEATURE_DIM), dtype=torch.float32)

        hetero["time"].x = unified_time
        hetero["time"].pos = torch.zeros((1, 2), dtype=torch.float32)
        hetero["time", "temporal", "time"].edge_index = torch.zeros((2, 0), dtype=torch.long)

        # 可选的跨源空间连接（保留）
        src_keys = [n.lower() for n in self.src_names]
        if "oco2" in src_keys and "cams" in src_keys:
            oco_key = "oco2_obs"
            cams_key = "cams_grid"
            if oco_key in hetero.node_types and cams_key in hetero.node_types:
                pos_src = hetero[oco_key].pos
                pos_tgt = hetero[cams_key].pos
                eidx = build_spatial_links(pos_src, pos_tgt, self.radius_deg, self.max_neighbors)
                hetero[oco_key, "aligned_with", cams_key].edge_index = eidx

        if "cams" in src_keys and "era5" in src_keys:
            cams_key = "cams_grid"
            era_key = "era5_grid"
            if cams_key in hetero.node_types and era_key in hetero.node_types:
                pos_src = hetero[cams_key].pos
                pos_tgt = hetero[era_key].pos
                eidx = build_spatial_links(pos_src, pos_tgt, self.radius_deg, self.max_neighbors)
                hetero[cams_key, "influenced_by", era_key].edge_index = eidx

        return hetero
