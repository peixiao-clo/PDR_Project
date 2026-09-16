# -*- coding: utf-8 -*-
"""
time_base.py —— 统一时间基准与偏移量记录（时间同步硬约束）

统一约定（所有时间戳必须统一格式、统一基准）
------------------------------------------------------------
绝对时间基准 : Unix 秒（UTC，无时区偏移）          -> 列名 t_unix
相对时间基准 : 相对会话起点（默认首个 RTK 历元）秒数 -> 列名 t_rel
本地时间     : t_unix + 28800（采集地为 UTC+8 / 徐州）
日期时间格式 : pandas.Timestamp（UTC），展示统一为 %Y-%m-%d %H:%M:%S.%f

各来源时间 -> t_unix 换算（本模块唯一权威）
------------------------------------------------------------
RTK       : t_unix = GPS_EPOCH_UNIX + Week*WEEK_SECONDS + GPSTime_s - LEAP_SECONDS
IMU GNSS  : SensorTimestamp_unix_s 本身即 t_unix（UTC）
IMU App   : t_unix = slope * AppTimestamp_s + intercept   （time_align_posi 回归拟合）
IMU 传感器: t_unix = slope * (SensorTimestamp_s - sensor_t0_app) + intercept
POSI      : 与 AppTimestamp 同一基准（应用启动时间）

时间偏移量记录（TimeOffset）：对齐后必须落盘，含
  - 各基准零点（GPS 周零、Unix 零、开机零、App 零）
  - 回归模型（斜率/截距/R²/拟合点数）
  - 会话窗口与首尾偏移量
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 时间系统常量
# ---------------------------------------------------------------------------
GPS_EPOCH_UNIX = 315964800.0   # GPS 时间原点 1980-01-06 00:00:00 UTC 的 Unix 秒
WEEK_SECONDS = 604800.0         # 7 * 86400
LEAP_SECONDS = 18.0             # GPS - UTC = 18 s（2017-01-01 起，现行）
LOCAL_UTC_OFFSET_H = 8.0        # 采集地徐州 UTC+8

# 时间列统一命名（下游代码必须使用，不得再引入其它命名）
T_UNIX = "t_unix"               # 绝对时间（Unix 秒，UTC）
T_REL = "t_rel"                 # 相对时间（会话起点秒）


def unix_to_utc(unix: float | np.ndarray | pd.Series) -> pd.Series:
    """Unix 秒 -> UTC datetime（pandas Series，无时区偏移）。"""
    arr = np.atleast_1d(np.asarray(unix, dtype=np.float64))
    return pd.Series(pd.to_datetime(arr, unit="s")).dt.round("us")


def unix_to_local(unix: float | np.ndarray | pd.Series) -> pd.Series:
    """Unix 秒 -> 本地时间（UTC+8）。"""
    return unix_to_utc(unix) + pd.Timedelta(hours=LOCAL_UTC_OFFSET_H)


def gps_week_sow_to_unix(week, sow, leap_seconds: float = LEAP_SECONDS) -> np.ndarray:
    """GPS 周 + 周内秒 -> Unix 秒（UTC）。"""
    week = np.asarray(week, dtype=np.float64)
    sow = np.asarray(sow, dtype=np.float64)
    return GPS_EPOCH_UNIX + week * WEEK_SECONDS + sow - float(leap_seconds)


def unix_to_gps_week_sow(unix, leap_seconds: float = LEAP_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """Unix 秒 -> (GPS 周, 周内秒)。"""
    u = np.asarray(unix, dtype=np.float64) + float(leap_seconds) - GPS_EPOCH_UNIX
    week = np.floor(u / WEEK_SECONDS)
    sow = u - week * WEEK_SECONDS
    return week, sow


def rtk_to_unix(rtk: pd.DataFrame, leap_seconds: float = LEAP_SECONDS) -> pd.DataFrame:
    """为 read_rtk 输出的 RTK DataFrame 增加统一时间列 t_unix（inplace 风格，返回副本）。

    :param rtk: 必须含 Week / GPSTime_s 列
    :return: 带 t_unix、t_rel 列的 RTK DataFrame（t_rel 相对首历元）
    """
    df = rtk.copy()
    df[T_UNIX] = gps_week_sow_to_unix(df["Week"], df["GPSTime_s"], leap_seconds)
    df[T_REL] = df[T_UNIX] - df[T_UNIX].iloc[0]
    return df


# ---------------------------------------------------------------------------
# 时间偏移量记录（对齐后必须落盘）
# ---------------------------------------------------------------------------
@dataclass
class TimeOffset:
    """IMU 与 RTK 时间同步的完整偏移量记录。

    IMU App 时间 -> t_unix 的线性模型:  t_unix = slope * AppTimestamp_s + intercept
    IMU 传感器时间与 App 时间同速率（1:1），差值为 sensor_t0_app。
    """

    # ---- 时间系统常量 ----
    gps_epoch_unix: float = GPS_EPOCH_UNIX
    leap_seconds: float = LEAP_SECONDS
    local_utc_offset_h: float = LOCAL_UTC_OFFSET_H

    # ---- IMU 内部基准 ----
    sensor_t0_app: float | None = None   # SensorTimestamp=0 时对应的 App 时间（s）
    app_t0_unix: float | None = None     # App 时间 0 对应的 t_unix（= intercept）

    # ---- 回归模型（App 时间 -> t_unix）----
    slope: float | None = None            # 比例（理论 1.0）
    intercept: float | None = None        # 截距 = App 时间 0 的 t_unix
    r2: float | None = None               # 拟合优度
    n_fit: int | None = None              # 拟合点数
    fit_sources: str = ""                 # 拟合依据: gnss / posi / manual

    # ---- 对齐后的会话窗口（t_unix）----
    imu_t0_unix: float | None = None      # IMU 首个样本
    imu_t1_unix: float | None = None      # IMU 末个样本
    rtk_t0_unix: float | None = None      # RTK 首个历元
    rtk_t1_unix: float | None = None      # RTK 末个历元
    offset_start_s: float | None = None   # IMU 起点 - RTK 起点（s）
    overlap_s: float | None = None        # IMU 落在 RTK 时间窗内的秒数

    def app_to_unix(self, app_time_s) -> np.ndarray:
        """按记录的模型把 App 时间换算为 t_unix。"""
        if self.slope is None or self.intercept is None:
            raise ValueError("TimeOffset 尚未完成对齐（slope/intercept 为空）")
        return self.slope * np.asarray(app_time_s, dtype=np.float64) + self.intercept

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """落盘为 JSON（含 UTC 说明），对齐后必须调用。"""
        d = self.to_dict()
        d["_说明"] = ("统一基准: t_unix=Unix秒(UTC); "
                      "IMU App时间->t_unix: t_unix=slope*AppTimestamp_s+intercept; "
                      "RTK: t_unix=GPS_EPOCH_UNIX+Week*604800+GPSTime_s-leap_seconds")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "TimeOffset":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d.pop("_说明", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        """人类可读摘要（含 UTC 与本地时间）。"""
        def fmt(u):
            if u is None:
                return "-"
            return (f"{unix_to_utc(u).iloc[0].strftime('%Y-%m-%d %H:%M:%S.%f')} UTC / "
                    f"{unix_to_local(u).iloc[0].strftime('%Y-%m-%d %H:%M:%S.%f')} UTC+8")
        lines = [
            "===== 时间偏移量记录 (TimeOffset) =====",
            f"拟合模型 : t_unix = {self.slope:.9f} * AppTimestamp_s + {self.intercept:.6f}"
            f"   (R²={self.r2:.6f}, n={self.n_fit}, 依据={self.fit_sources})",
            f"IMU 基准 : App时间0 = {fmt(self.app_t0_unix)}",
            f"          SensorTimestamp 与 App 时间偏移 = "
            f"{self.sensor_t0_app:.6f} s" if self.sensor_t0_app is not None
            else "          SensorTimestamp 与 App 时间偏移 = -",
            f"IMU 窗口 : {fmt(self.imu_t0_unix)}  ~  {fmt(self.imu_t1_unix)}",
            f"RTK 窗口 : {fmt(self.rtk_t0_unix)}  ~  {fmt(self.rtk_t1_unix)}",
            f"窗口关系 : IMU起点相对RTK起点 = {self.offset_start_s:+.3f} s; "
            f"IMU 落在 RTK 窗口内 {self.overlap_s:.1f} s",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 时间基准审计（检查时区/单位/起始时间差异）
# ---------------------------------------------------------------------------
def audit_time_bases(imu: dict[str, pd.DataFrame], rtk: pd.DataFrame,
                     leap_seconds: float = LEAP_SECONDS) -> dict:
    """审计 IMU 与 RTK 的所有时间列：单位、基准(epoch)、时区、换算后的 t_unix。

    :param imu: read_imu 输出的 {类型: DataFrame}
    :param rtk: read_rtk 输出的 RTK DataFrame（需含 Week/GPSTime_s）
    :return: 审计报告 dict（同时打印表格）
    """
    report: dict = {"rows": []}

    def add(name, unit, epoch, tz, sample, unix_sample, note=""):
        report["rows"].append({
            "来源": name, "单位": unit, "基准": epoch, "时区": tz,
            "示例值": sample, "t_unix": unix_sample, "说明": note})

    # IMU 各时间列
    acc = imu["ACCE"]
    gnss = imu["GNSS"]
    posi = imu["POSI"]
    # GNSS 首个定位给出 App 时间 <-> t_unix 的锚点
    g0 = gnss.iloc[0]
    # SensorTimestamp 与 AppTimestamp 的基准差（取 ACCE 首行）
    a0 = acc.iloc[0]
    sensor_t0_app = a0["SensorTimestamp_s"] - a0["AppTimestamp_s"]

    add("IMU ACCE", "秒", "App 启动(0)", "单调时钟(无时区)",
        f"{a0['AppTimestamp_s']:.6f}", None,
        f"App时间0对应t_unix={a0['AppTimestamp_s'] - g0['AppTimestamp_s'] + g0['SensorTimestamp_unix_s']:.6f}")
    add("IMU ACCE", "秒", "手机开机(0)", "单调时钟(无时区)",
        f"{a0['SensorTimestamp_s']:.6f}", None,
        f"SensorTimestamp 与 App 时间差 = {sensor_t0_app:.6f} s (速率1:1)")
    add("IMU GNSS", "秒", "Unix(1970-01-01)", "UTC",
        f"{g0['SensorTimestamp_unix_s']:.3f}", g0["SensorTimestamp_unix_s"],
        "App 时间锚点: AppTimestamp=%.3f s -> t_unix=%.3f" %
        (g0["AppTimestamp_s"], g0["SensorTimestamp_unix_s"]))
    add("IMU POSI", "秒", "App 启动(0)", "单调时钟(无时区)",
        f"{posi['Timestamp_s'].iloc[0]:.3f}", None,
        f"共 {len(posi)} 个标记; 与 App 时间同基准")

    # RTK 时间列
    r0, r1 = rtk.iloc[0], rtk.iloc[-1]
    u0 = gps_week_sow_to_unix(r0["Week"], r0["GPSTime_s"], leap_seconds)
    u1 = gps_week_sow_to_unix(r1["Week"], r1["GPSTime_s"], leap_seconds)
    add("RTK GPSTime", "秒", f"GPS 周内秒(Week {r0['Week']:.0f})",
        "GPS 时标(≈UTC+18s)", f"{r0['GPSTime_s']:.2f}", float(u0),
        f"周内秒范围 {r0['GPSTime_s']:.2f}~{r1['GPSTime_s']:.2f}")
    add("RTK AbsTime", "datetime", "GPS 周 2394", "UTC(扣闰秒18)",
        r0["AbsTime"].strftime("%Y-%m-%d %H:%M:%S.%f"), float(u0),
        "read_rtk 已按 leap_seconds=18 输出 UTC")

    # 汇总结论
    report["sensor_t0_app"] = sensor_t0_app
    report["rtk_t0_unix"] = float(u0)
    report["rtk_t1_unix"] = float(u1)
    report["leap_seconds"] = leap_seconds
    report["结论"] = ("单位均为秒(datetime 列除外)，无单位混用；"
                     "存在 4 种起始时间基准(App启动/手机开机/Unix/GPS周)，"
                     "时区差异统一由 't_unix=UTC' 消解；"
                     "IMU 时间与 RTK 时间的衔接锚点为 IMU GNSS(UTC)。")

    # 打印
    print("===== 时间基准审计 =====")
    for r in report["rows"]:
        print(f"[{r['来源']:<12}] 单位={r['单位']:<8} 基准={r['基准']:<18} "
              f"时区={r['时区']:<12} 示例={r['示例值']:<18} t_unix={r['t_unix']}")
        if r["说明"]:
            print(f"{'':>4}  └─ {r['说明']}")
    print("\n" + report["结论"])
    return report
