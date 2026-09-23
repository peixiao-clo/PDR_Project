# -*- coding: utf-8 -*-
"""
evaluate_all_to_excel.py —— 任务⑥：批量运行基线PDR，统计定位误差，输出Excel

流程：
  1. 遍历 5 个人员文件夹（曹/李好/刘杰/吴佳骏/吴懿婷）
  2. 每人 6 条 logfile_*.txt，按文件名日期自动配 RTK
  3. 复用 baseline_pdr_main 的函数跑完整 PDR 流程
  4. 统计：RMSE / 最大误差 / 75%分位数 / 终点误差(闭合误差B)
  5. 汇总输出 Excel

依赖：pandas, numpy, openpyxl
"""

import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 复用基线主程序（不要重复实现）
# ============================================================
import baseline_pdr_main as bp


# ============================================================
# 配置区（按需修改）
# ============================================================
DATA_ROOT = Path(r"D:\Learing\学校\大创—孙猛\模型搭建\程序搭建\数据文件")
RTK_DIR   = Path(r"D:\Learing\学校\大创—孙猛\模型搭建\程序搭建")
OUTPUT_XLSX = Path(r"D:\Learing\学校\大创—孙猛\模型搭建\程序搭建\output\baseline_accuracy_all.xlsx")

PERSONS = ["曹", "李好", "刘杰", "吴佳骏", "吴憶婷"]

# RTK 按日期索引（文件名规则：25号早上/下午、26号下午、27号）
RTK_BY_KEY = {
    "25-AM": RTK_DIR / "25号早上-329b.txt",
    "25-PM": RTK_DIR / "25号下午-329l.txt",
    "26":    RTK_DIR / "26号下午-330.txt",
    "27":    RTK_DIR / "27号-331.txt",
}

# 25号早/下午分界（小时）
AM_PM_HOUR = 12

# 临时输出目录（baseline_pdr_main 内部会写 CSV / png，这里隔离掉）
TMP_DIR = Path("./_tmp_eval")


# ============================================================
# 工具：从 IMU 文件名解析时间，选 RTK
# ============================================================
_IMU_RE = re.compile(r"logfile_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")


def pick_rtk(imu_path: Path) -> Path:
    """根据 IMU 文件名里的日期/时间，选对应 RTK 文件。"""
    m = _IMU_RE.search(imu_path.name)
    if not m:
        raise ValueError(f"IMU 文件名不符合规则: {imu_path.name}")

    y, mo, d, h, mi, s = map(int, m.groups())

    if d == 25:
        key = "25-AM" if h < AM_PM_HOUR else "25-PM"
    elif d == 26:
        key = "26"
    elif d == 27:
        key = "27"
    else:
        raise ValueError(f"未配置日期的 RTK 映射: {imu_path.name}（月={mo}, 日={d}）")

    rtk = RTK_BY_KEY[key]
    if not rtk.exists():
        raise FileNotFoundError(f"RTK 文件不存在: {rtk}")
    return rtk


def percentile75(errors: np.ndarray) -> float:
    return float(np.percentile(errors, 75))


# ============================================================
# 单条轨迹：跑 PDR + 算指标
# ============================================================
def run_one(imu_path: Path, rtk_path: Path) -> dict:
    """
    跑一条轨迹。返回 dict（指标 + 明细 DataFrame）。
    异常由调用方捕获。
    """
    # 改写 baseline_pdr_main 的全局变量（其函数直接读这些全局）
    bp.IMU_FILE = str(imu_path)
    bp.RTK_FILE = str(rtk_path)
    # 隔离输出，避免 30 次覆盖同一份 CSV/png
    bp.OUTPUT_DIR = TMP_DIR / imu_path.stem
    bp.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1~4 读取 + 截取 ----
    data, acc_df, gyro_df, ahrs_df, posi_df = bp.load_imu(str(imu_path))
    rtk_df = bp.load_rtk(str(rtk_path))
    acc_seg, gyro_seg, ahrs_seg = bp.extract_walking_segment(
        acc_df, gyro_df, ahrs_df, posi_df
    )
    acc_time = acc_seg["AppTimestamp_s"].to_numpy(dtype=float)
    acc_data, gyro_data, ahrs_data = bp.adapt_columns(
        acc_seg, gyro_seg, ahrs_seg
    )

    # ---- 5~7 PDR 核心 ----
    peaks, step_lengths, headings = bp.run_pdr(
        acc_data, gyro_data, ahrs_data
    )
    if len(peaks) == 0:
        raise ValueError("步态检测失败，步数为 0")

    # ---- 8 位置递推 ----
    positions = bp.compute_positions(peaks, step_lengths, headings)

    # ---- 9 时间对齐 + 误差 ----
    rtk_enu_aligned, info = bp.align_rtk_to_steps(
        positions, peaks, acc_time, data, rtk_df
    )
    metrics = bp.evaluate(positions, rtk_enu_aligned, info)

    # ---- 任务⑥ 指标（补 75%分位；闭合误差=B 终点误差） ----
    errors = np.asarray(metrics["errors"], dtype=float)
    pdr_xy = np.asarray(metrics["pdr_xy"], dtype=float)
    rtk_xy = np.asarray(metrics["rtk_xy"], dtype=float)

    closure_error = float(np.linalg.norm(pdr_xy[-1] - rtk_xy[-1]))  # B 定义

    row = {
        "person":          imu_path.parent.name,
        "imu_file":        imu_path.name,
        "rtk_file":        rtk_path.name,
        "n_steps_detected": int(len(peaks)),
        "n_steps_used":    int(metrics["n_used"]),
        "n_steps_dropped": int(metrics["n_dropped"]),
        "rmse_m":          float(metrics["rmse"]),
        "mean_error_m":    float(metrics["mean_error"]),
        "max_error_m":     float(metrics["max_error"]),
        "p75_error_m":     percentile75(errors),
        "closure_error_m": closure_error,
        "avg_step_len_m":  float(np.mean(step_lengths)) if len(step_lengths) else np.nan,
        "align_r2":        float(info["r2"]),
        "overlap_s":       float(info["overlap_s"]),
        "match_method":    info["method"],
        "max_gap_s":       float(info["max_gap_s"]),
        "dt_median_s":     float(info["dt_median"]) if np.isfinite(info["dt_median"]) else np.nan,
        "dt_max_s":        float(info["dt_max"]) if np.isfinite(info["dt_max"]) else np.nan,
    }

    detail = pd.DataFrame({
        "step_index": np.arange(len(errors)),
        "E_pdr_m": pdr_xy[:, 0],
        "N_pdr_m": pdr_xy[:, 1],
        "E_rtk_m": rtk_xy[:, 0],
        "N_rtk_m": rtk_xy[:, 1],
        "error_m": errors,
    })

    return {"row": row, "detail": detail}


# ============================================================
# 批量主流程
# ============================================================
def main():
    print("=" * 70)
    print("任务⑥ 批量运行基线PDR + 统计定位误差")
    print("=" * 70)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"数据根目录不存在: {DATA_ROOT}")
    if not RTK_DIR.exists():
        raise FileNotFoundError(f"RTK 目录不存在: {RTK_DIR}")

    all_rows = []
    all_details = []          # 每个元素 = (person, imu_name, DataFrame)
    failed = []               # 失败记录

    for person in PERSONS:
        pdir = DATA_ROOT / person
        if not pdir.exists():
            print(f"⚠️ 跳过（文件夹不存在）: {pdir}")
            continue

        imu_files = sorted(pdir.glob("logfile_*.txt"))
        print(f"\n【{person}】 找到 {len(imu_files)} 条轨迹")

        for imu_path in imu_files:
            try:
                rtk_path = pick_rtk(imu_path)
                print(f"  ▶ {imu_path.name}")
                print(f"     配 RTK: {rtk_path.name}")

                res = run_one(imu_path, rtk_path)
                all_rows.append(res["row"])
                all_details.append((person, imu_path.name, res["detail"]))

                print(f"     ✓ RMSE={res['row']['rmse_m']:.2f}m, "
                      f"Max={res['row']['max_error_m']:.2f}m, "
                      f"P75={res['row']['p75_error_m']:.2f}m, "
                      f"终点={res['row']['closure_error_m']:.2f}m, "
                      f"步数={res['row']['n_steps_used']}")

            except Exception as e:
                print(f"     ✗ 失败: {e}")
                traceback.print_exc()
                failed.append({
                    "person":   person,
                    "imu_file": imu_path.name,
                    "error":    str(e),
                })

    if not all_rows:
        print("\n❌ 没有任何轨迹成功，退出。")
        return

    df_summary = pd.DataFrame(all_rows)

    # ---- 按人统计（6 条轨迹的均值/最大/最小） ----
    metric_cols = ["rmse_m", "mean_error_m", "max_error_m",
                   "p75_error_m", "closure_error_m"]
    by_person = (
        df_summary.groupby("person")[metric_cols]
        .agg(["mean", "max", "min"])
    )
    # 展平列名：rmse_m_mean / rmse_m_max / ...
    by_person.columns = [f"{c}_{s}" for c, s in by_person.columns]
    by_person = by_person.reset_index()

    # ---- 总统计（全部 30 条） ----
    total_row = {f"{c}_mean": float(df_summary[c].mean()) for c in metric_cols}
    total_row.update({f"{c}_max":  float(df_summary[c].max())  for c in metric_cols})
    total_row.update({f"{c}_min":  float(df_summary[c].min())  for c in metric_cols})
    total_row.update({f"{c}_std":  float(df_summary[c].std())  for c in metric_cols})
    df_total = pd.DataFrame([total_row])

    # ---- 明细：合并所有轨迹（加 person / imu 两列） ----
    detail_frames = []
    for person, imu_name, d in all_details:
        dd = d.copy()
        dd.insert(0, "imu_file", imu_name)
        dd.insert(0, "person", person)
        detail_frames.append(dd)
    df_detail = pd.concat(detail_frames, ignore_index=True)

    # ---- 失败记录 ----
    df_failed = pd.DataFrame(failed) if failed else pd.DataFrame(
        columns=["person", "imu_file", "error"]
    )

    # ---- 写 Excel ----
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n写入 Excel: {OUTPUT_XLSX}")

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="汇总_每条轨迹", index=False)
        by_person.to_excel(writer,  sheet_name="按人统计",     index=False)
        df_total.to_excel(writer,   sheet_name="总统计",       index=False)
        df_detail.to_excel(writer,  sheet_name="每步明细",     index=False, float_format="%.6f")
        if len(df_failed) > 0:
            df_failed.to_excel(writer, sheet_name="失败记录", index=False)

    # ---- 控制台打印摘要 ----
    print("\n" + "=" * 70)
    print("总体指标（全部 {} 条轨迹）".format(len(df_summary)))
    print("=" * 70)
    for c in metric_cols:
        print(f"  {c:<20s} mean={df_summary[c].mean():.3f}  "
              f"max={df_summary[c].max():.3f}  "
              f"min={df_summary[c].min():.3f}")

    print("\n按人（RMSE / 最大误差 / 75%分位 / 终点误差）:")
    for _, r in by_person.iterrows():
        print(f"  {r['person']:<6s}  "
              f"RMSE={r['rmse_m_mean']:.3f}  "
              f"Max={r['max_error_m_mean']:.3f}  "
              f"P75={r['p75_error_m_mean']:.3f}  "
              f"终点={r['closure_error_m_mean']:.3f}")

    if len(df_failed) > 0:
        print(f"\n⚠️ 有 {len(df_failed)} 条轨迹失败，详见 Excel「失败记录」表：")
        for _, r in df_failed.iterrows():
            print(f"  {r['person']} / {r['imu_file']}: {r['error']}")

    print("\n✅ 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()