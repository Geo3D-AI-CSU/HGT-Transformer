# check_oco.py
"""
OCO-2 数据完整性检测脚本
---------------------------------
功能：
1. 检查 OCO-2 数据目录中 .nc4 文件的存在性与可读性；
2. 自动解析日期（根据文件名 oco2_LtCO2_YYMMDD_...nc4）；
3. 检查文件内关键变量 ['latitude', 'longitude', 'xco2'] 是否存在；
4. 输出：
   - 正常文件数量
   - 缺失日期
   - 损坏文件（无法读取或变量缺失）

用法：
  python check_oco.py D:/GCN-Transformer/data/OCO-2
"""

import os
import re
import sys
import datetime as dt
import netCDF4 as nc

def extract_date_from_filename(filename):
    """
    从 OCO-2 文件名提取日期，例如：
    oco2_LtCO2_200101_B11100Ar_230603192102s.nc4 -> datetime(2020, 1, 1)
    """
    match = re.search(r'oco2_LtCO2_(\d{6})_', filename)
    if not match:
        return None
    datestr = match.group(1)
    year = int('20' + datestr[:2])
    month = int(datestr[2:4])
    day = int(datestr[4:6])
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def check_oco_file(filepath):
    """
    尝试读取 OCO-2 文件并检测关键变量是否存在
    """
    try:
        with nc.Dataset(filepath, 'r') as ds:
            for var in ['latitude', 'longitude', 'xco2']:
                if var not in ds.variables:
                    return False, f"缺少变量 {var}"
            # 检查是否为空
            lat = ds.variables['latitude'][:]
            lon = ds.variables['longitude'][:]
            xco2 = ds.variables['xco2'][:]
            if lat.size == 0 or lon.size == 0 or xco2.size == 0:
                return False, "数据为空"
        return True, "正常"
    except Exception as e:
        return False, str(e)


def scan_oco_directory(oco_dir):
    """
    扫描 OCO-2 目录下的所有 .nc4 文件并检测
    """
    print(f"📂 正在扫描目录: {oco_dir}")
    if not os.path.exists(oco_dir):
        print("❌ 目录不存在，请检查路径！")
        return

    files = sorted([f for f in os.listdir(oco_dir) if f.endswith('.nc4')])
    if not files:
        print("❌ 未发现任何 .nc4 文件！")
        return

    print(f"📁 共检测到 {len(files)} 个文件。")
    ok_files, bad_files = [], []
    dates_detected = []

    for f in files:
        path = os.path.join(oco_dir, f)
        date = extract_date_from_filename(f)
        if date:
            dates_detected.append(date)
        ok, msg = check_oco_file(path)
        if ok:
            ok_files.append((f, date))
        else:
            bad_files.append((f, date, msg))

    # 获取日期范围
    if dates_detected:
        start = min(dates_detected)
        end = max(dates_detected)
        all_days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
        missing_dates = [d for d in all_days if d not in dates_detected]
    else:
        start = end = None
        missing_dates = []

    print("\n📊 检查结果汇总")
    print("──────────────────────────────")
    print(f"✅ 正常文件: {len(ok_files)}")
    print(f"⚠️  损坏文件: {len(bad_files)}")
    print(f"❌ 缺失日期: {len(missing_dates)}")

    if start and end:
        print(f"📅 数据日期范围: {start} ～ {end}")

    if missing_dates:
        print("\n缺失日期示例:", [d.strftime("%Y-%m-%d") for d in missing_dates[:10]], "...")
    if bad_files:
        print("\n损坏文件示例:")
        for f, d, msg in bad_files[:5]:
            date_str = d.strftime("%Y-%m-%d") if d else "未知日期"
            print(f" - {f} ({date_str}): {msg}")

    print("\n✅ 检查完成。")


if __name__ == "__main__":
    oco_dir = "D:/GCN-Transformer/data/OCO-2"
    scan_oco_directory(oco_dir)
