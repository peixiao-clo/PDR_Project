# -*- coding: utf-8 -*-
"""
read_rtk.py —— RTK 参考轨迹读取脚本（Inertial Explorer / Waypoint 输出）

任务②（第 1 周 · 截止 9.9）
============================================================
功能
  - 解析 Inertial Explorer 导出的 GNSS/INS 历元文件（空格/制表符分隔）
  - Week + GPSTime → 绝对时间（GPS 时标，可选闰秒修正为 UTC）→ 相对时间(s)
  - 纬度/经度 DMS（度 分 秒）→ 十进制度数
  - 输入：文件路径；输出：DataFrame（含 ECEF、经纬度、椭高、速度、绝对/相对时间）

文件格式（头部约 12 行元信息/表头/单位行，数据自其后开始）
      Week   GPSTime   X-ECEF   Y-ECEF   Z-ECEF
   Latitude(D M S)  Longitude(D M S)  H-Ell
   VX-ECEF  VY-ECEF  VZ-ECEF  VelBdyX  VelBdyY  VelBdyZ
  （合计 18 列，见 RAW_COLUMNS）

说明
  - GPSTime 为 GPS 周内秒（s），Week 为 GPS 周数；GPS 时间原点 1980-01-06 00:00:00
  - AbsTime 默认按 GPS 时标输出（不扣闰秒）；传入 leap_seconds=18 可得到 UTC
  - RelTime_s 为相对指定参考历元（默认文件首历元）的时间（s）

用法
  from read_rtk import read_rtk
  rtk = read_rtk('25号早上-329b.txt')
  rtk[['Week', 'GPSTime_s', 'Lat_deg', 'Lon_deg', 'AbsTime', 'RelTime_s']].head()

命令行
  python read_rtk.py 25号早上-329b.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GPS_EPOCH = pd.Timestamp("1980-01-06 00:00:00")
WEEK_SECONDS = 7 * 86400

# 18 个原始列（空格分隔）；其中 Latitude/Longitude 各占 3 列（D M S）
RAW_COLUMNS = [
    "Week", "GPSTime_s",
    "X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m",
    "LatD", "LatM", "LatS",
    "LonD", "LonM", "LonS",
    "H_Ell_m",
    "VX_ECEF_mps", "VY_ECEF_mps", "VZ_ECEF_mps",
    "VelBdyX_mps", "VelBdyY_mps", "VelBdyZ_mps",
]


def dms_to_deg(d: float, m: float, s: float) -> float:
    """度分秒 → 十进制度（符号取自已度分量）。"""
    sign = -1.0 if d < 0 else 1.0
    return sign * (abs(float(d)) + float(m) / 60.0 + float(s) / 3600.0)


def dms_to_deg_series(d: pd.Series, m: pd.Series, s: pd.Series) -> pd.Series:
    """向量化 DMS → 十进制度。"""
    sign = np.where(d < 0, -1.0, 1.0)
    return sign * (d.abs() + m / 60.0 + s / 3600.0)


def week_sow_to_absolute(week, sow, leap_seconds: int | None = None) -> pd.Series:
    """Week + GPSTime(s) → 绝对时间。

    :param week: GPS 周（int/float 或 Series）
    :param sow:  GPS 周内秒（float 或 Series）
    :param leap_seconds: 闰秒数；为 None 时输出 GPS 时标（不修正），
                         传 18 则修正为 UTC（GPS - 18 = UTC）
    :return: pd.Series（datetime64[ns]，UTC 时区无偏移）
    """
    week = np.asarray(week, dtype=np.float64)
    sow = np.asarray(sow, dtype=np.float64)
    seconds = week * WEEK_SECONDS + sow
    if leap_seconds is not None:
        seconds = seconds - float(leap_seconds)
    return pd.Series(GPS_EPOCH + pd.to_timedelta(seconds, unit="s"))


def week_sow_to_relative(week, sow, reference: tuple[float, float] | None = None,
                         leap_seconds: int | None = None) -> pd.Series:
    """Week + GPSTime(s) → 相对时间（相对参考历元，默认文件首历元）。

    :param week: GPS 周（int/float 或 Series）
    :param sow:  GPS 周内秒
    :param reference: (ref_week, ref_sow)；None 时取第一个历元
    :param leap_seconds: 与 week_sow_to_absolute 一致（修正与否不影响相对差）
    :return: pd.Series，单位为秒
    """
    abs_series = week_sow_to_absolute(week, sow, leap_seconds=leap_seconds)
    if reference is None:
        t0 = abs_series.iloc[0]
    else:
        t0 = week_sow_to_absolute(float(reference[0]), float(reference[1]),
                                  leap_seconds=leap_seconds).iloc[0]
    return (abs_series - t0).dt.total_seconds()


def _find_data_start(lines: list[str]) -> int:
    """定位数据起始行（跳过元信息、表头行与单位行）。"""
    for i, line in enumerate(lines):
        toks = line.split()
        if toks and toks[0] == "Week":
            return i + 2  # 跳过表头行 + 单位行
    raise ValueError("未找到 'Week' 表头行，请确认是否为 Inertial Explorer 导出格式")


def read_rtk(path: str | Path, reference: tuple[float, float] | None = None,
             leap_seconds: int | None = None) -> pd.DataFrame:
    """读取 RTK / PPP 参考轨迹文件。

    :param path: 文件路径
    :param reference: (ref_week, ref_sow) 相对时间参考；None 表示文件首历元
    :param leap_seconds: 闰秒修正（None=GPS 时标；18=UTC），默认 None
    :return: DataFrame，列为
        Week, GPSTime_s, X/Y/Z_ECEF_m, Lat_deg, Lon_deg, H_Ell_m,
        VX/VY/VZ_ECEF_mps, VelBdyX/Y/Z_mps, AbsTime, RelTime_s
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    data_start = _find_data_start(lines)

    df = pd.read_csv(
        path, sep=r"\s+", skiprows=data_start, header=None,
        names=RAW_COLUMNS, dtype=np.float64,
    )
    # DMS → 十进制度数
    df["Lat_deg"] = dms_to_deg_series(df["LatD"], df["LatM"], df["LatS"])
    df["Lon_deg"] = dms_to_deg_series(df["LonD"], df["LonM"], df["LonS"])
    # 绝对时间与相对时间
    df["AbsTime"] = week_sow_to_absolute(df["Week"], df["GPSTime_s"],
                                         leap_seconds=leap_seconds).dt.round("us")
    df["RelTime_s"] = week_sow_to_relative(df["Week"], df["GPSTime_s"],
                                           reference=reference,
                                           leap_seconds=leap_seconds)
    # 删除原始 DMS 列，整理列序
    df = df.drop(columns=["LatD", "LatM", "LatS", "LonD", "LonM", "LonS"])
    cols = (["Week", "GPSTime_s", "AbsTime", "RelTime_s"] +
            [c for c in df.columns if c not in
             ("Week", "GPSTime_s", "AbsTime", "RelTime_s")])
    return df[cols]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTK 参考轨迹读取（Inertial Explorer 格式）")
    parser.add_argument("file", help="IE 导出的历元文件路径")
    parser.add_argument("--leap", type=int, default=None,
                        help="闰秒数（如 18 得到 UTC），默认按 GPS 时标输出")
    parser.add_argument("--out", default=None,
                        help="导出 CSV 路径（自动附加 t_unix/t_rel 统一时间列）")
    args = parser.parse_args()

    rtk = read_rtk(args.file, leap_seconds=args.leap)
    print(f"历元数: {len(rtk)}")
    print(f"时间范围: {rtk['AbsTime'].iloc[0]} ~ {rtk['AbsTime'].iloc[-1]} "
          f"(UTC+8: {(rtk['AbsTime'].iloc[0] + pd.Timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')}"
          f" ~ {(rtk['AbsTime'].iloc[-1] + pd.Timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"相对时间范围: {rtk['RelTime_s'].iloc[0]:.1f} ~ {rtk['RelTime_s'].iloc[-1]:.1f} s "
          f"({(rtk['RelTime_s'].iloc[-1] - rtk['RelTime_s'].iloc[0]) / 60:.1f} min)")
    print("\n前 5 行（关键列）:")
    print(rtk[["Week", "GPSTime_s", "X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m",
               "Lat_deg", "Lon_deg", "H_Ell_m", "RelTime_s"]].head().to_string(index=False))

    if args.out:
        from time_base import rtk_to_unix
        df = rtk_to_unix(rtk, leap_seconds=(args.leap if args.leap is not None else 18.0))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, float_format="%.6f", encoding="utf-8-sig")
        print(f"\n[导出] 已保存: {args.out}（含统一时间列 t_unix/t_rel，leap={args.leap or 18}）")
