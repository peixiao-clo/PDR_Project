# -*- coding: utf-8 -*-
"""
read_imu.py —— 手机 IMU 数据批量读取脚本（GetSensorData App 格式）

任务①（第 1 周 · 截止 9.8）
============================================================
功能
  - 批量解析 GetSensorData App（LOPSI 研究组 / CAR-CSIC，西班牙）导出的
    手机传感器日志，按传感器类型分类解析为 pandas.DataFrame
  - 自动跳过 '%' 注释头、空行与字段数不匹配的非法行（统计并报告）
  - 输入：单个文件路径 或 路径列表；输出：{传感器类型: DataFrame} 字典

文件格式（分号分隔，'%' 开头为注释行）
  ACCE;AppTimestamp(s);SensorTimestamp(s);Acc_X(m/s^2);Acc_Y(m/s^2);Acc_Z(m/s^2);Accuracy(int)
  GYRO;AppTimestamp(s);SensorTimestamp(s);Gyr_X(rad/s);Gyr_Y(rad/s);Gyr_Z(rad/s);Accuracy(int)
  MAGN;AppTimestamp(s);SensorTimestamp(s);Mag_X(uT);Mag_Y(uT);Mag_Z(uT);Accuracy(int)
  AHRS;AppTimestamp(s);SensorTimestamp(s);PitchX(deg);RollY(deg);YawZ(deg);Quat(2);Quat(3);Quat(4);Accuracy(int)
  PRES;AppTimestamp(s);SensorTimestamp(s);Pres(mbar);Accuracy(int)
  LIGH;AppTimestamp(s);SensorTimestamp(s);Light(lux);Accuracy(int)
  PROX;AppTimestamp(s);SensorTimestamp(s);prox(?);Accuracy(int)
  GNSS;AppTimestamp(s);SensorTimestamp(unix,s);Lat(deg);Lon(deg);Alt(m);Bearing(deg);Acc(m);Speed(m/s);SatInView;SatInUse
  POSI;Timestamp(s);Counter;Latitude(deg);Longitude(deg);FloorID;BuildingID
  WIFI;AppTimestamp(s);SensorTimestamp(s);SSID;BSSID;Freq(MHz);RSS(dBm)
  SOUN;AppTimestamp(s);RMS;Pressure(Pa);SPL(dB)

说明
  - AppTimestamp 与 SensorTimestamp 单位均为秒（SensorTimestamp 为传感器
    自开机时间，相对差值与 AppTimestamp 一致，可用于采样率估计）
  - GNSS 的 SensorTimestamp 为 Unix 时间戳（秒，UTC）
  - 本项目主要使用 ACCE / GYRO / AHRS / POSI，其余类型一并解析备用

用法
  from read_imu import read_imu
  data = read_imu('曹.txt')            # 单文件
  data = read_imu(['a.txt', 'b.txt'])  # 批量
  acc = data['ACCE']                   # DataFrame

命令行
  python read_imu.py 曹.txt [更多文件...]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from time_base import TimeOffset, T_UNIX, T_REL

# ---------------------------------------------------------------------------
# 各传感器类型的列定义（列顺序与文件格式严格一致）
# ---------------------------------------------------------------------------
SENSOR_SCHEMAS: dict[str, tuple[str, ...]] = {
    "ACCE": ("AppTimestamp_s", "SensorTimestamp_s",
             "Acc_X_mps2", "Acc_Y_mps2", "Acc_Z_mps2", "Accuracy"),
    "GYRO": ("AppTimestamp_s", "SensorTimestamp_s",
             "Gyr_X_radps", "Gyr_Y_radps", "Gyr_Z_radps", "Accuracy"),
    "MAGN": ("AppTimestamp_s", "SensorTimestamp_s",
             "Mag_X_uT", "Mag_Y_uT", "Mag_Z_uT", "Accuracy"),
    "AHRS": ("AppTimestamp_s", "SensorTimestamp_s",
             "Pitch_deg", "Roll_deg", "Yaw_deg",
             "Quat2", "Quat3", "Quat4", "Accuracy"),
    "PRES": ("AppTimestamp_s", "SensorTimestamp_s", "Pressure_mbar", "Accuracy"),
    "LIGH": ("AppTimestamp_s", "SensorTimestamp_s", "Light_lux", "Accuracy"),
    "PROX": ("AppTimestamp_s", "SensorTimestamp_s", "Prox", "Accuracy"),
    "TEMP": ("AppTimestamp_s", "SensorTimestamp_s", "Temp_C", "Accuracy"),
    "HUMI": ("AppTimestamp_s", "SensorTimestamp_s", "Humi_pct", "Accuracy"),
    "GNSS": ("AppTimestamp_s", "SensorTimestamp_unix_s",
             "Lat_deg", "Lon_deg", "Alt_m", "Bearing_deg",
             "Accuracy_m", "Speed_mps", "SatInView", "SatInUse"),
    "POSI": ("Timestamp_s", "Counter", "Lat_deg", "Lon_deg",
             "FloorID", "BuildingID"),
    "WIFI": ("AppTimestamp_s", "SensorTimestamp_s",
             "SSID", "BSSID", "Freq_MHz", "RSS_dBm"),
    "SOUN": ("AppTimestamp_s", "RMS", "Pressure_Pa", "SPL_dB"),
}

# 文本列（不做数值强制转换）
TEXT_COLUMNS: frozenset[str] = frozenset({"SSID", "BSSID"})


def _parse_one_file(path: Path) -> tuple[dict[str, list[list[str]]], int]:
    """解析单个文件，返回 {类型: 原始字符串行列表} 与非法行计数。

    :param path: 传感器日志文件路径
    :return: (rows_dict, bad_rows)
    """
    rows: dict[str, list[list[str]]] = defaultdict(list)
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split(";")
            cat = parts[0]
            fields = parts[1:]
            schema = SENSOR_SCHEMAS.get(cat)
            if schema is None:
                bad += 1
                continue  # 未知类型，跳过并计数
            # 容错：移除多余空字段（如行尾 ';' 或文档中的双分号占位）
            if len(fields) > len(schema) and "" in fields:
                cleaned = [x for x in fields if x != ""]
                if len(cleaned) == len(schema):
                    fields = cleaned
            if len(fields) == len(schema):
                rows[cat].append(fields)
            else:
                bad += 1
    return rows, bad


def _to_dataframe(rows: list[list[str]], schema: tuple[str, ...]) -> pd.DataFrame:
    """将某类型的原始行列表转为 DataFrame，并做数值强制转换。"""
    df = pd.DataFrame(rows, columns=list(schema))
    for col in df.columns:
        if col not in TEXT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_imu(paths: str | Path | list[str] | list[Path],
             verbose: bool = False) -> dict[str, pd.DataFrame]:
    """批量读取手机 IMU 数据。

    :param paths:  单个文件路径或路径列表（str / Path 均可）
    :param verbose: 是否打印每个文件的解析统计
    :return: {传感器类型: DataFrame} 字典；多文件时同类型数据按序拼接，
             通过 'SourceFile' 列保留来源文件名
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    merged: dict[str, list[pd.DataFrame]] = defaultdict(list)
    total_bad = 0
    for p in paths:
        path = Path(p)
        rows, bad = _parse_one_file(path)
        total_bad += bad
        if verbose:
            print(f"[read_imu] {path.name}: "
                  f"{sum(len(v) for v in rows.values())} 行有效, {bad} 行跳过")
        for cat, cat_rows in rows.items():
            df = _to_dataframe(cat_rows, SENSOR_SCHEMAS[cat])
            df.insert(0, "SourceFile", path.name)
            merged[cat].append(df)

    result: dict[str, pd.DataFrame] = {}
    for cat, dfs in merged.items():
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("AppTimestamp_s").reset_index(drop=True) \
            if "AppTimestamp_s" in df.columns else df
        result[cat] = df

    if verbose:
        print(f"[read_imu] 累计跳过非法行: {total_bad}")
    return result


def summarize(data: dict[str, pd.DataFrame]) -> str:
    """生成数据集摘要文本（行数 / 时间范围 / 估计采样率）。"""
    lines = [f"{'类型':<6}{'行数':>9}{'App时间范围(s)':>22}{'估计采样率':>12}"]
    for cat in sorted(data):
        df = data[cat]
        if "AppTimestamp_s" not in df.columns:
            lines.append(f"{cat:<6}{len(df):>9}{'(POSI/GNSS 特殊)':>22}")
            continue
        t = df["AppTimestamp_s"].dropna()
        n = len(t)
        if n > 1:
            dt = t.diff().dropna()
            dt = dt[(dt > 0) & (dt < 1.0)]  # 剔除停顿/大间隔
            rate = f"{1.0 / dt.median():.1f} Hz" if len(dt) else "N/A"
            span = f"{t.min():.2f} ~ {t.max():.2f}"
        else:
            rate, span = "N/A", "-"
        lines.append(f"{cat:<6}{n:>9}{span:>22}{rate:>12}")
    return "\n".join(lines)


def add_unified_time(df: pd.DataFrame, offset: TimeOffset | None) -> pd.DataFrame:
    """附加统一时间列 t_unix / t_rel（time_base 唯一权威）。

    - GNSS : SensorTimestamp_unix_s 本身即 t_unix（UTC）
    - 其余 : AppTimestamp_s / POSI Timestamp_s 经 TimeOffset 模型换算
    - offset 为 None 时不加列（避免伪造默认值）
    """
    if offset is None:
        return df.copy()
    out = df.copy()
    if "SensorTimestamp_unix_s" in out.columns:
        out[T_UNIX] = pd.to_numeric(out["SensorTimestamp_unix_s"], errors="coerce")
    elif "AppTimestamp_s" in out.columns:
        out[T_UNIX] = offset.app_to_unix(out["AppTimestamp_s"])
    elif "Timestamp_s" in out.columns:  # POSI 与 App 时间同基准
        out[T_UNIX] = offset.app_to_unix(out["Timestamp_s"])
    if T_UNIX in out.columns and offset.rtk_t0_unix is not None:
        out[T_REL] = out[T_UNIX] - offset.rtk_t0_unix
    return out


def export_csv(data: dict[str, pd.DataFrame], out_dir: str | Path,
               prefix: str = "imu", offset: TimeOffset | None = None,
               keep_source: bool = False) -> Path:
    """把 read_imu 结果按类型导出为统一 CSV（数据整理落盘）。

    - <out_dir>/<prefix>_<TYPE>.csv    每类型一个文件，列名统一
    - <out_dir>/<prefix>_manifest.csv  汇总清单（类型/行数/文件/是否含 t_unix）
    - 传入 TimeOffset（time_align 输出）时自动附加 t_unix/t_rel 统一时间列
    - keep_source=False（默认）：CSV 不含 SourceFile 列（来源信息在 manifest）；
      keep_source=True：批量多文件时保留，便于区分各会话
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cat in sorted(data):
        df = add_unified_time(data[cat], offset)
        if not keep_source and "SourceFile" in df.columns:
            df = df.drop(columns=["SourceFile"])
        path = out_dir / f"{prefix}_{cat}.csv"
        df.to_csv(path, index=False, float_format="%.6f", encoding="utf-8-sig")
        manifest.append({"Type": cat, "Rows": len(df), "File": path.name,
                         "Has_t_unix": T_UNIX in df.columns})
    mpath = out_dir / f"{prefix}_manifest.csv"
    pd.DataFrame(manifest).to_csv(mpath, index=False, encoding="utf-8-sig")
    return mpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手机 IMU 数据批量读取（GetSensorData 格式）")
    parser.add_argument("files", nargs="+", help="一个或多个传感器日志文件路径")
    parser.add_argument("--out", default=None,
                        help="导出目录：按类型输出 <dir>/<stem>_<TYPE>.csv + manifest")
    parser.add_argument("--offset", default=None,
                        help="time_offset.json 路径（存在则自动附加 t_unix/t_rel 统一时间列）")
    parser.add_argument("--keep-source", action="store_true",
                        help="保留 SourceFile 列（默认删除；批量多文件时建议保留）")
    args = parser.parse_args()

    data = read_imu(args.files, verbose=True)
    print("\n========== 数据集摘要 ==========")
    print(summarize(data))
    if "POSI" in data:
        print("\n---- POSI 参考标记 ----")
        print(data["POSI"].to_string(index=False))

    if args.out:
        offset = TimeOffset.from_json(args.offset) if args.offset else None
        prefix = Path(args.files[0]).stem
        mpath = export_csv(data, args.out, prefix=prefix, offset=offset,
                           keep_source=args.keep_source)
        print(f"\n[导出] 完成 → {args.out}/（清单: {mpath.name}）")
        if args.keep_source:
            print("[导出] 已保留 SourceFile 列（--keep-source）")
        else:
            print("[导出] 已删除 SourceFile 列（批量多文件需区分会话时加 --keep-source）")
        if offset is not None:
            print(f"[导出] 已附加统一时间列 {T_UNIX}/{T_REL} "
                  f"（依据 {args.offset} 的偏移模型）")
        else:
            print(f"[导出] 未附加统一时间列——用 --offset 指定 time_offset.json "
                  f"可自动换算（time_align_posi 运行后生成）")
