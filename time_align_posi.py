# -*- coding: utf-8 -*-
"""
time_align_posi.py —— 利用 POSI 标记 / GNSS 锚点实现 IMU 与 RTK 时间粗对齐（线性回归）

任务④（第 1 周 · 截止 9.12）
============================================================
功能
  - 建立 IMU 时间（AppTimestamp_s，与 POSI 同基准）到统一绝对时间
    t_unix（Unix 秒，UTC）的线性模型：t_unix = slope * AppTimestamp_s + intercept
  - 拟合依据（按优先级自动选择）：
      1) IMU 自带 GNSS 行：AppTimestamp_s <-> SensorTimestamp_unix_s(=t_unix) 成对数据
         （本数据 POSI 经纬度全为 0，无法按位置匹配，GNSS 为唯一可用锚点）
      2) POSI 位置标记：若 POSI 带真实经纬度，按最近邻匹配到 RTK 轨迹取 t_unix
      3) 手动指定 (app_t0_unix) 兜底
  - 输出 TimeOffset（斜率+截距+R²+拟合点数+各基准零点+会话窗口），落盘 JSON
  - 对齐后自动校验：IMU 会话窗口与 RTK 窗口的重叠情况

用法
  from read_imu import read_imu
  from read_rtk import read_rtk
  from time_align_posi import align
  imu, rtk = read_imu('曹.txt'), read_rtk('25号早上-329b.txt')
  offset = align(imu, rtk, out_json='outputs/time_offset.json')

命令行
  python time_align_posi.py 曹.txt 25号早上-329b.txt [--out 输出.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from time_base import (TimeOffset, LEAP_SECONDS, gps_week_sow_to_unix,
                       unix_to_utc, unix_to_local, audit_time_bases)


# ---------------------------------------------------------------------------
# 回归拟合
# ---------------------------------------------------------------------------
def _linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    """一阶线性回归，返回 (slope, intercept, R², n)。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2:
        raise ValueError(f"拟合点数不足: {n}")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2, n


def fit_alignment_gnss(imu: dict[str, pd.DataFrame],
                       leap_seconds: float = LEAP_SECONDS) -> TimeOffset:
    """依据 IMU 自带 GNSS 行拟合 App 时间 -> t_unix 的线性模型。

    GNSS 行同时携带 AppTimestamp_s（App 时间）与 SensorTimestamp_unix_s（UTC Unix 秒），
    二者即 (x, y) 成对样本；重复定位（相同 t_unix）去重后参与回归。
    """
    gnss = imu["GNSS"].copy()
    gnss = gnss.dropna(subset=["AppTimestamp_s", "SensorTimestamp_unix_s"])
    gnss = gnss.drop_duplicates(subset=["SensorTimestamp_unix_s"])  # 同一历元只计一次
    x = gnss["AppTimestamp_s"].to_numpy(dtype=np.float64)
    y = gnss["SensorTimestamp_unix_s"].to_numpy(dtype=np.float64)

    slope, intercept, r2, n = _linreg(x, y)

    off = TimeOffset(leap_seconds=leap_seconds)
    off.slope, off.intercept, off.r2, off.n_fit = slope, intercept, r2, n
    off.app_t0_unix = intercept
    off.fit_sources = "gnss"
    # SensorTimestamp 与 App 时间同速率(1:1)，基准差取 ACCE 首行
    acc0 = imu["ACCE"].iloc[0]
    off.sensor_t0_app = float(acc0["SensorTimestamp_s"] - acc0["AppTimestamp_s"])
    return off


def fit_alignment_posi(imu: dict[str, pd.DataFrame], rtk: pd.DataFrame,
                       leap_seconds: float = LEAP_SECONDS,
                       max_dist_m: float = 20.0) -> TimeOffset:
    """依据 POSI 标记按位置最近邻匹配到 RTK 轨迹，拟合 App 时间 -> t_unix。

    要求 POSI 行含真实经纬度；若全部为 0（本数据现状）则抛出提示。
    """
    posi = imu["POSI"]
    if posi["Lat_deg"].abs().sum() == 0 and posi["Lon_deg"].abs().sum() == 0:
        raise ValueError("POSI 经纬度全为 0，无法按位置匹配；请使用 fit_alignment_gnss")

    # RTK 位置列 -> 经纬度，计算 POSI 与每个 RTK 历元的平面距离（简化等距圆柱）
    lat_r = rtk["Lat_deg"].to_numpy()
    lon_r = rtk["Lon_deg"].to_numpy()
    t_r = gps_week_sow_to_unix(rtk["Week"], rtk["GPSTime_s"], leap_seconds)
    cos_lat = np.cos(np.deg2rad(lat_r.mean()))

    xs, ys = [], []
    for _, p in posi.iterrows():
        dlat = (lat_r - p["Lat_deg"]) * 111320.0
        dlon = (lon_r - p["Lon_deg"]) * 111320.0 * cos_lat
        dist = np.hypot(dlat, dlon)
        k = int(np.argmin(dist))
        if dist[k] > max_dist_m:
            continue  # 无匹配（距离超限）则跳过
        xs.append(p["Timestamp_s"])
        ys.append(t_r[k])

    if len(xs) < 2:
        raise ValueError(f"POSI 有效匹配点不足 2 个（{len(xs)}），无法回归")

    slope, intercept, r2, n = _linreg(np.array(xs), np.array(ys))
    off = TimeOffset(leap_seconds=leap_seconds)
    off.slope, off.intercept, off.r2, off.n_fit = slope, intercept, r2, n
    off.app_t0_unix = intercept
    off.fit_sources = "posi"
    return off


# ---------------------------------------------------------------------------
# 对齐主流程
# ---------------------------------------------------------------------------
def _fill_window(off: TimeOffset, imu: dict[str, pd.DataFrame],
                 rtk: pd.DataFrame, leap_seconds: float) -> None:
    """填充会话窗口与重叠校验。"""
    acc = imu["ACCE"]
    t0 = off.app_to_unix(acc["AppTimestamp_s"].iloc[0])
    t1 = off.app_to_unix(acc["AppTimestamp_s"].iloc[-1])
    off.imu_t0_unix, off.imu_t1_unix = float(t0), float(t1)
    off.rtk_t0_unix = float(gps_week_sow_to_unix(
        rtk["Week"].iloc[0], rtk["GPSTime_s"].iloc[0], leap_seconds))
    off.rtk_t1_unix = float(gps_week_sow_to_unix(
        rtk["Week"].iloc[-1], rtk["GPSTime_s"].iloc[-1], leap_seconds))
    off.offset_start_s = off.imu_t0_unix - off.rtk_t0_unix
    overlap = min(off.imu_t1_unix, off.rtk_t1_unix) - max(off.imu_t0_unix, off.rtk_t0_unix)
    off.overlap_s = max(0.0, overlap)


def align(imu: dict[str, pd.DataFrame], rtk: pd.DataFrame,
          method: str = "auto", leap_seconds: float = LEAP_SECONDS,
          out_json: str | Path | None = None,
          verbose: bool = True) -> TimeOffset:
    """IMU 与 RTK 时间粗对齐（线性回归），返回并（可选）落盘 TimeOffset。

    :param imu:   read_imu 输出
    :param rtk:   read_rtk 输出（含 Week/GPSTime_s）
    :param method: 'auto'(gnss 优先，失败转 posi) / 'gnss' / 'posi'
    :param out_json: 偏移量 JSON 落盘路径（对齐后必须记录，推荐 outputs/time_offset.json）
    :return: TimeOffset
    """
    if method in ("auto", "gnss"):
        try:
            off = fit_alignment_gnss(imu, leap_seconds)
        except Exception as e:
            if method == "gnss":
                raise
            off = fit_alignment_posi(imu, rtk, leap_seconds)
    elif method == "posi":
        off = fit_alignment_posi(imu, rtk, leap_seconds)
    else:
        raise ValueError(f"未知 method: {method}")

    _fill_window(off, imu, rtk, leap_seconds)
    if out_json is not None:
        off.to_json(out_json)
    if verbose:
        print(off.summary())
        print(f"[time_align] 偏移量已记录 -> {out_json}" if out_json else "")
    return off


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMU 与 RTK 时间粗对齐（任务④）")
    parser.add_argument("imu_file", help="IMU 日志文件")
    parser.add_argument("rtk_file", help="RTK 历元文件")
    parser.add_argument("--method", choices=["auto", "gnss", "posi"], default="auto")
    parser.add_argument("--out", default="outputs/time_offset.json",
                        help="偏移量 JSON 输出路径")
    args = parser.parse_args()

    from read_imu import read_imu
    from read_rtk import read_rtk

    imu_data = read_imu(args.imu_file)
    rtk_data = read_rtk(args.rtk_file)
    print(f"[time_align] IMU 类型: {sorted(imu_data)} ; RTK 历元: {len(rtk_data)}\n")

    audit_time_bases(imu_data, rtk_data)
    print()
    align(imu_data, rtk_data, method=args.method, out_json=args.out)
