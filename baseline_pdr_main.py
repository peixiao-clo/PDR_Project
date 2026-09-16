# -*- coding: utf-8 -*-
"""
baseline_pdr_main.py —— 传统PDR基线主程序（时间对齐版）

流程：
  1. 读取IMU数据（read_imu.py）
  2. 读取RTK数据（read_rtk.py）
  3. 截取有效行走段（POSI标记）
  4. 列名适配
  5. 步态检测（baseline_pdr.step_detection）
  6. 步长估计（baseline_pdr.step_length_weinberg）
  7. 航向估计（baseline_pdr.heading_complementary_filter）
  8. 位置递推（baseline_pdr.position_update）
  9. 时间对齐（time_align_posi.align + match_rtk）→ 误差评估
  10. 可视化输出

关键：误差评估按“每步时刻”对 RTK 插值对齐，而非按行号硬截。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from read_imu import read_imu
from read_rtk import read_rtk
from ecef_to_enu import ecef_to_enu
from time_base import rtk_to_unix, TimeOffset
from time_align_posi import align
from match_rtk import match_rtk
from baseline_pdr import (
    step_detection,
    step_length_weinberg,
    heading_complementary_filter,
    position_update,
)

# ============================================================
# 配置区
# ============================================================
IMU_FILE = "数据文件/曹/logfile_2025_11_25_09_19_53.txt"
RTK_FILE = "25号早上-329b.txt"
FS = 100
OUTPUT_DIR = Path("output")

# PDR参数
PEAK_HEIGHT = 2.0
PEAK_DISTANCE = 20
K_WEINBERG = 0.45
ALPHA_COMP = 0.90            # 互补滤波：陀螺仪权重（0.98→0.90，抑制漂移）
REMOVE_GYRO_BIAS = True      # 是否扣除 gyro_z 零偏

# 时间对齐参数
MATCH_METHOD = "interp"      # 'interp'（线性插值，推荐）或 'nearest'
MAX_GAP_S = 0.5              # 查询时刻距最近 RTK 历元的最大允许时间差
OFFSET_JSON = OUTPUT_DIR / "time_offset.json"


# ============================================================
# 中文字体设置（避免 SimHei 缺字形警告）
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 读取IMU数据
# ============================================================
def load_imu(imu_file):
    """读取IMU数据，返回加速度/陀螺仪/AHRS/POSI"""
    print(f"[1/9] 读取IMU数据: {imu_file}")
    data = read_imu(imu_file)

    acc_df = data['ACCE']
    gyro_df = data['GYRO']
    ahrs_df = data['AHRS']
    posi_df = data.get('POSI', pd.DataFrame())

    print(f"  ACCE: {len(acc_df)} 行")
    print(f"  GYRO: {len(gyro_df)} 行")
    print(f"  AHRS: {len(ahrs_df)} 行")
    print(f"  POSI: {len(posi_df)} 行")

    return data, acc_df, gyro_df, ahrs_df, posi_df


# ============================================================
# 2. 读取RTK数据
# ============================================================
def load_rtk(rtk_file):
    """读取RTK数据"""
    print(f"[2/9] 读取RTK数据: {rtk_file}")
    rtk_df = read_rtk(rtk_file)
    print(f"  RTK: {len(rtk_df)} 行")
    return rtk_df


# ============================================================
# 3. 截取有效行走段（POSI标记）
# ============================================================
def extract_walking_segment(acc_df, gyro_df, ahrs_df, posi_df):
    """
    根据POSI标记截取有效行走段。

    POSI标记顺序：
      POSI 1: 第一次静止标定结束
      POSI 2: 八字旋转标定结束 → 行走开始
      POSI 3: 行走结束 → 第二次静止标定开始
      POSI 4: 第二次八字旋转标定结束
    """
    print(f"[3/9] 截取有效行走段")

    if len(posi_df) < 4:
        print(f"  ⚠️ POSI标记不足4个（实际{len(posi_df)}个），使用全部数据")
        return acc_df, gyro_df, ahrs_df

    t_start = posi_df.iloc[1]['Timestamp_s']
    t_end = posi_df.iloc[2]['Timestamp_s']

    print(f"  行走段: {t_start:.2f}s ~ {t_end:.2f}s (时长{t_end - t_start:.1f}s)")

    def _slice(df):
        return df[(df['AppTimestamp_s'] >= t_start) &
                  (df['AppTimestamp_s'] <= t_end)].reset_index(drop=True)

    acc_seg = _slice(acc_df)
    gyro_seg = _slice(gyro_df)
    ahrs_seg = _slice(ahrs_df)

    print(f"  截取后: ACCE={len(acc_seg)}, GYRO={len(gyro_seg)}, AHRS={len(ahrs_seg)}")

    # 长度一致性检查（position_update 里用 peaks 索引 headings，长度必须一致）
    n = min(len(acc_seg), len(gyro_seg), len(ahrs_seg))
    if not (len(acc_seg) == len(gyro_seg) == len(ahrs_seg)):
        print(f"  ⚠️ ACCE/GYRO/AHRS 长度不一致，统一截到 {n} 行")
        acc_seg = acc_seg.iloc[:n].reset_index(drop=True)
        gyro_seg = gyro_seg.iloc[:n].reset_index(drop=True)
        ahrs_seg = ahrs_seg.iloc[:n].reset_index(drop=True)

    return acc_seg, gyro_seg, ahrs_seg


# ============================================================
# 4. 数据格式适配（列名转换）
# ============================================================
def adapt_columns(acc_df, gyro_df, ahrs_df):
    """将 read_imu 输出的列名适配为 baseline_pdr 期望的格式。"""
    print(f"[4/9] 列名适配")

    acc_data = {
        'acc_x': acc_df['Acc_X_mps2'].values,
        'acc_y': acc_df['Acc_Y_mps2'].values,
        'acc_z': acc_df['Acc_Z_mps2'].values,
    }
    gyro_data = {
        'gyro_x': gyro_df['Gyr_X_radps'].values,
        'gyro_y': gyro_df['Gyr_Y_radps'].values,
        'gyro_z': gyro_df['Gyr_Z_radps'].values,
    }
    ahrs_data = {
        'yaw': ahrs_df['Yaw_deg'].values,
        'pitch': ahrs_df['Pitch_deg'].values,
        'roll': ahrs_df['Roll_deg'].values,
    }

    print(f"  ACCE: {len(acc_data['acc_x'])} 点")
    print(f"  GYRO: {len(gyro_data['gyro_z'])} 点")
    print(f"  AHRS: {len(ahrs_data['yaw'])} 点")

    return acc_data, gyro_data, ahrs_data


# ============================================================
# 5-7. PDR核心
# ============================================================
def run_pdr(acc_data, gyro_data, ahrs_data):
    """执行PDR三大模块：步态检测 → 步长估计 → 航向估计"""
    print(f"[5/9] 步态检测")
    peaks, intervals = step_detection(
        acc_data, fs=FS,
        peak_height=PEAK_HEIGHT,
        peak_distance=PEAK_DISTANCE
    )
    print(f"  检测到步数: {len(peaks)}")
    if len(intervals) > 0:
        print(f"  平均步间间隔: {np.mean(intervals):.1f} 采样点 "
              f"({np.mean(intervals) / FS:.2f}s)")

    if len(peaks) == 0:
        print(f"  ⚠️ 步数为0，降低peak_height至1.0重试...")
        peaks, intervals = step_detection(acc_data, fs=FS, peak_height=1.0,
                                          peak_distance=PEAK_DISTANCE)
        print(f"  重试后检测到步数: {len(peaks)}")

    if len(peaks) == 0:
        print(f"  ⚠️ 步数仍为0，降低至0.5重试...")
        peaks, intervals = step_detection(acc_data, fs=FS, peak_height=0.5,
                                          peak_distance=15)
        print(f"  重试后检测到步数: {len(peaks)}")

    print(f"[6/9] 步长估计")
    step_lengths = step_length_weinberg(acc_data, peaks, fs=FS, K=K_WEINBERG)
    if len(step_lengths) > 0:
        print(f"  平均步长: {np.mean(step_lengths):.3f} m")
        print(f"  步长范围: {np.min(step_lengths):.3f} ~ {np.max(step_lengths):.3f} m")

    print(f"[7/9] 航向估计")
    headings = heading_complementary_filter(
        gyro_data, ahrs_data, fs=FS,
        alpha=ALPHA_COMP, remove_bias=REMOVE_GYRO_BIAS
    )
    h_norm = np.degrees(headings)
    h_norm = (h_norm + 180) % 360 - 180
    print(f"  航向范围(归一后): {np.min(h_norm):.1f}° ~ {np.max(h_norm):.1f}°")

    return peaks, step_lengths, headings


# ============================================================
# 8. 位置递推
# ============================================================
def compute_positions(peaks, step_lengths, headings):
    """位置递推"""
    print(f"[8/9] 位置递推")
    positions = position_update(peaks, step_lengths, headings)
    print(f"  位置点数: {len(positions)}")
    print(f"  终点位置: ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f})")
    return positions


# ============================================================
# 9. 时间对齐 + 误差评估
# ============================================================
def align_rtk_to_steps(positions, peaks, acc_time, imu_data, rtk_df):
    """
    把 RTK 按“每步时刻”插值对齐，返回与 positions 逐行对应的 RTK ENU。

    :param positions: (N,2) PDR 每步位置
    :param peaks:     步态事件在 acc_seg 中的采样点索引
    :param acc_time:  acc_seg['AppTimestamp_s'].values，长度与 acc_seg 一致
    :param imu_data:  read_imu 输出的完整 dict（供 align 拟合偏移量）
    :param rtk_df:    read_rtk 输出
    :return: (rtk_enu_aligned: DataFrame, info: dict)
    """
    print(f"[9/9] 时间对齐 + 误差评估")

    # 9.1 拟合 IMU App 时间 -> t_unix 的线性模型
    off = align(imu_data, rtk_df, method="auto",
                out_json=OFFSET_JSON, verbose=False)
    print(f"  时间对齐: R²={off.r2:.6f}, n_fit={off.n_fit}, "
          f"IMU∩RTK 重叠={off.overlap_s:.1f}s")

    # 9.2 RTK 加统一时间列 t_unix
    rtk_u = rtk_to_unix(rtk_df)

    # 9.3 每步时刻（t_unix）：peaks 索引基准 = acc_seg，acc_time 同源
    n_steps = len(positions)
    step_peaks = peaks[:n_steps]
    assert len(step_peaks) == n_steps, \
        f"peaks 与 positions 长度不匹配: {len(step_peaks)} vs {n_steps}"

    t_q = off.app_to_unix(acc_time[step_peaks])   # 每步的 t_unix

    # 9.4 用每步时刻对 RTK 插值。
    #     fields 必须覆盖 ecef_to_enu 内部要取的 9 列 + rtk_to_unix 需要的 Week/GPSTime_s
    fields = ["Week", "GPSTime_s",
              "X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m",
              "Lat_deg", "Lon_deg", "H_Ell_m",
              "VX_ECEF_mps", "VY_ECEF_mps", "VZ_ECEF_mps"]
    matched = match_rtk(rtk_u, t_q, method=MATCH_METHOD,
                        fields=fields, max_gap_s=MAX_GAP_S)

    n_gap = matched.attrs.get("n_gap", 0)
    dt_src = matched["dt_source_s"].to_numpy()
    print(f"  match_rtk: method={MATCH_METHOD}, max_gap_s={MAX_GAP_S}")
    print(f"  gap(超窗/超限置 NaN): {n_gap} / {len(matched)} 步")
    if np.isfinite(dt_src).any():
        print(f"  dt_source_s: 中位={np.nanmedian(dt_src):.4f}s, "
              f"最大={np.nanmax(dt_src):.4f}s")

    # 9.5 匹配到的 ECEF -> ENU
    rtk_enu_aligned = ecef_to_enu(matched, leap_seconds=18.0)

    info = {
        "n_gap": n_gap,
        "dt_median": float(np.nanmedian(dt_src)) if np.isfinite(dt_src).any() else np.nan,
        "dt_max": float(np.nanmax(dt_src)) if np.isfinite(dt_src).any() else np.nan,
        "r2": off.r2,
        "overlap_s": off.overlap_s,
        "method": MATCH_METHOD,
        "max_gap_s": MAX_GAP_S,
    }
    return rtk_enu_aligned, info


def evaluate(positions, rtk_enu_aligned, info):
    """
    误差评估：PDR 每步位置 vs 同一步时刻的 RTK ENU 插值点。
    剔除 match_rtk 产生的 NaN 行，并报告剔除步数。
    """
    pdr = np.asarray(positions, dtype=float)[:, :2]
    rtk = np.asarray(rtk_enu_aligned[["E_m", "N_m"]].to_numpy(), dtype=float)

    assert len(pdr) == len(rtk), f"PDR 与 RTK 行数不一致: {len(pdr)} vs {len(rtk)}"

    # 剔除 NaN 行
    valid = np.isfinite(pdr).all(axis=1) & np.isfinite(rtk).all(axis=1)
    n_dropped = int((~valid).sum())
    pdr_v, rtk_v = pdr[valid], rtk[valid]
    if n_dropped:
        print(f"  ⚠️ 剔除 {n_dropped} 步（RTK 匹配为 NaN），剩余 {len(pdr_v)} 步参与评估")
    else:
        print(f"  全部 {len(pdr_v)} 步参与评估（无 NaN）")

    if len(pdr_v) == 0:
        raise ValueError("没有有效步参与误差评估，请检查时间对齐 / RTK 时间窗")

    # 公共原点：匹配后 RTK 的第一步位置
    origin = rtk_v[0].copy()
    pdr_v = pdr_v - origin
    rtk_v = rtk_v - origin

    errors = np.linalg.norm(pdr_v - rtk_v, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    closure_error = float(np.linalg.norm(pdr_v[-1] - rtk_v[-1]))

    print(f"  RMSE: {rmse:.2f} m")
    print(f"  平均误差: {mean_error:.2f} m")
    print(f"  最大误差: {max_error:.2f} m")
    print(f"  终点误差: {closure_error:.2f} m")

    return {
        "rmse": rmse,
        "mean_error": mean_error,
        "max_error": max_error,
        "closure_error": closure_error,
        "errors": errors,
        "n_dropped": n_dropped,
        "n_used": len(pdr_v),
        "pdr_xy": pdr_v,
        "rtk_xy": rtk_v,
    }


# ============================================================
# 10. 可视化
# ============================================================
def visualize(metrics, rtk_enu_aligned, output_dir):
    """
    轨迹对比 + 误差曲线。
    - 蓝实线：RTK 原始连续轨迹（背景参考）
    - 黑点  ：每步对应的 RTK 插值点
    - 红虚线：PDR 逐步位置
    """
    print(f"[可视化] 生成图表")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdr = metrics["pdr_xy"]
    rtk_steps = metrics["rtk_xy"]
    errors = metrics["errors"]

    # 原始 RTK 连续轨迹（用于背景），以步对齐后的首点为原点
    rtk_cont = rtk_enu_aligned[["E_m", "N_m"]].to_numpy(dtype=float)
    rtk_cont = rtk_cont - rtk_steps[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 图1：轨迹对比
    axes[0].plot(rtk_cont[:, 0], rtk_cont[:, 1], 'b-', linewidth=1.5,
                 alpha=0.6, label='RTK原始轨迹')
    axes[0].plot(rtk_steps[:, 0], rtk_steps[:, 1], 'k.', markersize=4,
                 label='每步RTK插值点')
    axes[0].plot(pdr[:, 0], pdr[:, 1], 'r--', linewidth=2, label='PDR估计')
    axes[0].plot(0, 0, 'go', markersize=12, label='公共原点')
    axes[0].set_xlabel('东向 (m)')
    axes[0].set_ylabel('北向 (m)')
    axes[0].set_title('轨迹对比（时间对齐）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # 图2：误差曲线
    axes[1].plot(errors, 'r-', linewidth=1.5)
    axes[1].axhline(y=np.mean(errors), color='b', linestyle='--',
                    label=f'平均误差={np.mean(errors):.2f}m')
    axes[1].set_xlabel('步数')
    axes[1].set_ylabel('定位误差 (m)')
    axes[1].set_title('误差随步数变化（时间对齐）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "baseline_pdr_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {save_path}")
    plt.show()


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("传统PDR基线主程序（时间对齐版）")
    print("=" * 60)

    # 1. 读取IMU
    data, acc_df, gyro_df, ahrs_df, posi_df = load_imu(IMU_FILE)

    # 2. 读取RTK
    rtk_df = load_rtk(RTK_FILE)

    # 3. 截取行走段
    acc_seg, gyro_seg, ahrs_seg = extract_walking_segment(acc_df, gyro_df, ahrs_df, posi_df)

    # IMU 采样时间轴（与 acc_seg 同长，供 peaks 索引）
    acc_time = acc_seg["AppTimestamp_s"].to_numpy(dtype=float)

    # 4. 列名适配
    acc_data, gyro_data, ahrs_data = adapt_columns(acc_seg, gyro_seg, ahrs_seg)

    # 5-7. PDR核心
    peaks, step_lengths, headings = run_pdr(acc_data, gyro_data, ahrs_data)

    if len(peaks) == 0:
        print("\n❌ 步态检测失败，无法继续。请检查数据或调整参数。")
        return

    # 8. 位置递推
    positions = compute_positions(peaks, step_lengths, headings)

    # 9. 时间对齐 + 误差评估
    rtk_enu_aligned, info = align_rtk_to_steps(positions, peaks, acc_time, data, rtk_df)
    metrics = evaluate(positions, rtk_enu_aligned, info)

    # 10. 可视化
    visualize(metrics, rtk_enu_aligned, OUTPUT_DIR)

    # 保存误差统计
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_df = pd.DataFrame([{
        "RMSE_m": metrics["rmse"],
        "MeanError_m": metrics["mean_error"],
        "MaxError_m": metrics["max_error"],
        "ClosureError_m": metrics["closure_error"],
        "StepCount": len(peaks),
        "StepsUsed": metrics["n_used"],
        "StepsDropped": metrics["n_dropped"],
        "AvgStepLength_m": float(np.mean(step_lengths)) if len(step_lengths) > 0 else 0.0,
        "AlphaComp": ALPHA_COMP,
        "RemoveGyroBias": REMOVE_GYRO_BIAS,
        "MatchMethod": info["method"],
        "MaxGap_s": info["max_gap_s"],
        "dtMedian_s": info["dt_median"],
        "dtMax_s": info["dt_max"],
        "AlignR2": info["r2"],
        "Overlap_s": info["overlap_s"],
    }])
    stats_path = output_dir / "baseline_metrics.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"\n误差统计已保存: {stats_path}")

    # 每步对齐明细（供溯源）
    n = len(metrics["pdr_xy"])
    matched_detail = pd.DataFrame({
        "step_index": np.arange(n),
        "E_pdr_m": metrics["pdr_xy"][:, 0],
        "N_pdr_m": metrics["pdr_xy"][:, 1],
        "E_rtk_m": metrics["rtk_xy"][:, 0],
        "N_rtk_m": metrics["rtk_xy"][:, 1],
        "error_m": metrics["errors"],
    })
    detail_path = output_dir / "matched_rtk_steps.csv"
    matched_detail.to_csv(detail_path, index=False,
                          float_format="%.6f", encoding="utf-8-sig")
    print(f"每步对齐明细已保存: {detail_path}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()