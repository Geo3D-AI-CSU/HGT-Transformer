import os
import xarray as xr

# 数据路径
tccon_dir = "D:/GCN-Transformer/data/TCCON"

# 仅选择合肥（hf开头）和香河（xh开头）的文件
files = [f for f in os.listdir(tccon_dir) if
         f.lower().endswith(".nc") and (f.lower().startswith("hf") or f.lower().startswith("xh"))]

if not files:
    print("❌ 未找到以 hf 或 xh 开头的 TCCON 文件，请检查文件命名！")
else:
    print(f"✅ 共找到 {len(files)} 个 TCCON 文件：")
    for f in files:
        print(" -", f)

    print("\n======================== 文件详情 ========================")
    for f in files:
        path = os.path.join(tccon_dir, f)
        print(f"\n📂 文件: {f}")
        try:
            ds = xr.open_dataset(path)

            # 打印维度信息
            print("  ➤ 维度 (dimensions):", dict(ds.dims))

            # 打印变量名称
            print("  ➤ 变量 (variables):", list(ds.data_vars))

            # 对每个变量打印详细信息
            for var in ds.data_vars:
                v = ds[var]
                unit = v.attrs.get("units", "未知单位")
                print(f"     • {var}: shape={v.shape}, dtype={v.dtype}, 单位={unit}")

            # 打印时间信息
            if "time" in ds.coords:
                times = ds["time"].values
                if len(times) > 0:
                    print(f"  ➤ 时间范围: {str(times[0])[:10]} → {str(times[-1])[:10]}")
                else:
                    print("  ⚠️ 未检测到时间数据")
            else:
                print("  ⚠️ 文件中没有时间维度")

            ds.close()
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
