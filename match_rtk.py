# -*- coding: utf-8 -*-
"""
match_rtk.py —— RTK 与 IMU 匹配：最近邻 / 插值，显式选择，禁止静默复制

============================================================
规则（硬约束）
  - 匹配方法必须显式传入：method='nearest' 或 'interp'，不设隐式默认
  - 'nearest'：零阶保持，把最近 RTK 历元的整行状态赋给查询时刻，
              不产生新数值；结果行与源历元一一对应（可视为复制，但显式标注 dt）
  - 'interp' ：一阶线性插值，在 RTK 相邻历元间按查询时刻内插出新数值
              （10Hz RTK -> 50Hz IMU 时必然产生新值）；插值仅对数值列进行
  - 查询时刻超出 RTK 时间窗：输出 NaN 并计数（gap），不伪造、不外推
  - 结果包含 dt_source_s（与最近源历元的时间差）与 is_exact（=最近邻 / 插值中恰好落在历元上）

用法
  from read_imu import read_imu
  from read_rtk import read_rtk
  from time_align_posi import align
  from time_base import rtk_to_unix
  from match_rtk import match_rtk
  imu, rtk = read_imu('曹.txt'), read_rtk('25号早上-329b.txt')
  off = align(imu, rtk, out_json='outputs/time_offset.json', verbose=False)
  rtk_u = rtk_to_unix(rtk)                      # 统一时间列 t_unix
  t_q = off.app_to_unix(imu['ACCE']['AppTimestamp_s'])   # IMU 采样时刻(unix)
  matched = match_rtk(rtk_u, t_q, method='interp')       # 显式选择
  matched[['t_unix','dt_source_s','is_exact','Lat_deg','Lon_deg']].head()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from time_base import T_UNIX

# 参与匹配的数值列（RTK 全部状态量）
POSITION_FIELDS = ["X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m", "Lat_deg", "Lon_deg", "H_Ell_m"]
VELOCITY_FIELDS = ["VX_ECEF_mps", "VY_ECEF_mps", "VZ_ECEF_mps",
                   "VelBdyX_mps", "VelBdyY_mps", "VelBdyZ_mps"]
DEFAULT_FIELDS = POSITION_FIELDS + VELOCITY_FIELDS


def match_rtk(rtk: pd.DataFrame, t_query,
              method: str, fields: list[str] | None = None,
              max_gap_s: float = 0.5) -> pd.DataFrame:
    """在查询时刻 t_query（t_unix，与 RTK 同基准）上取 RTK 状态。

    :param rtk:     必须含 t_unix 列（用 time_base.rtk_to_unix 生成），并按 t_unix 升序
    :param t_query: 查询时刻（标量或数组，t_unix）
    :param method:  'nearest'（最近邻，零阶保持）或 'interp'（线性插值）——必填
    :param fields:  需要匹配的数值列；默认位置+速度全部状态
    :param max_gap_s: 查询时刻与最近源历元的最大允许时间差；超过则该行置 NaN 并计入 gap
    :return: DataFrame（索引与 t_query 对齐）：
        t_unix, 各 fields, dt_source_s（距最近源历元秒）, is_exact（是否正好落在历元）
    """
    if method not in ("nearest", "interp"):
        raise ValueError(f"method 必须为 'nearest' 或 'interp'，收到: {method!r}")
    if T_UNIX not in rtk.columns:
        raise KeyError("rtk 缺少 t_unix 列，请先用 time_base.rtk_to_unix() 生成统一时间列")
    if fields is None:
        fields = list(DEFAULT_FIELDS)

    rtk = rtk.sort_values(T_UNIX).reset_index(drop=True)
    t_src = rtk[T_UNIX].to_numpy(dtype=np.float64)
    t_q = np.atleast_1d(np.asarray(t_query, dtype=np.float64))

    # 最近源历元索引
    idx = np.searchsorted(t_src, t_q)
    idx = np.clip(idx, 1, len(t_src) - 1)
    left, right = idx - 1, idx
    dt_left = np.abs(t_q - t_src[left])
    dt_right = np.abs(t_src[right] - t_q)
    nearest = np.where(dt_left <= dt_right, left, right)
    dt_src = np.minimum(dt_left, dt_right)

    out = pd.DataFrame({T_UNIX: t_q})
    out["dt_source_s"] = dt_src
    out["is_exact"] = dt_src < 1e-6

    for col in fields:
        if col not in rtk.columns:
            raise KeyError(f"rtk 缺少列: {col}")
        vals = rtk[col].to_numpy(dtype=np.float64)
        if method == "nearest":
            out[col] = vals[nearest]
        else:  # interp
            out[col] = np.interp(t_q, t_src, vals)

    # 超窗 / 超限处理：不伪造、不外推
    gap = (t_q < t_src[0] - max_gap_s) | (t_q > t_src[-1] + max_gap_s) | (dt_src > max_gap_s)
    n_gap = int(gap.sum())
    if n_gap:
        out.loc[gap, fields] = np.nan
        out.loc[gap, "dt_source_s"] = np.nan
    out.attrs["method"] = method
    out.attrs["n_gap"] = n_gap
    return out


def compare_methods(rtk: pd.DataFrame, t_query, fields=None) -> dict:
    """对比最近邻与插值在位置量上的差异，辅助明确选择匹配方法。"""
    fields = fields or ["X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m", "Lat_deg", "Lon_deg"]
    n = match_rtk(rtk, t_query, method="nearest", fields=fields)
    i = match_rtk(rtk, t_query, method="interp", fields=fields)
    stats = {}
    for col in fields:
        d = (n[col] - i[col]).abs().dropna()
        stats[col] = {"mean": float(d.mean()), "max": float(d.max())}
    return {"n_nearest": len(n), "n_interp": len(i), "fields": stats}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RTK 与 IMU 匹配（最近邻/插值）")
    parser.add_argument("imu_file")
    parser.add_argument("rtk_file")
    parser.add_argument("--method", choices=["nearest", "interp"], required=True,
                        help="匹配方法（必填，禁止默认）")
    parser.add_argument("--offset", default=None,
                        help="已有的 time_offset.json 路径（优先使用，不重新 align）")
    parser.add_argument("--out", default=None,
                        help="导出匹配结果 CSV 路径（含 t_unix + RTK 状态量 + dt_source_s + is_exact）")
    args = parser.parse_args()

    from read_imu import read_imu
    from read_rtk import read_rtk
    from time_base import rtk_to_unix, TimeOffset

    imu = read_imu(args.imu_file, verbose=False)
    rtk = read_rtk(args.rtk_file)

    # 优先使用已有偏移量；未指定则现场 align（需要 IMU GNSS 锚点）
    if args.offset:
        off = TimeOffset.from_json(args.offset)
        print(f"[match_rtk] 使用已有偏移量: {args.offset}")
    else:
        from time_align_posi import align
        off = align(imu, rtk, out_json=None, verbose=False)
        print("[match_rtk] 现场计算偏移量（未指定 --offset）")

    rtk_u = rtk_to_unix(rtk)
    t_q = off.app_to_unix(imu["ACCE"]["AppTimestamp_s"].to_numpy())

    m = match_rtk(rtk_u, t_q, method=args.method)
    print(f"[match_rtk] 方法={args.method}, 查询 {len(t_q)} 个 IMU 采样时刻")
    print(f"  gap(超窗/超限置 NaN): {m.attrs['n_gap']} 行")
    print(m[["t_unix", "dt_source_s", "is_exact", "Lat_deg", "Lon_deg", "H_Ell_m"]]
          .head(8).round(4).to_string(index=False))

    print("\n[对比] 最近邻 vs 插值 位置差异（辅助决策）:")
    for col, s in compare_methods(rtk_u, t_q)["fields"].items():
        print(f"  {col:<12} 均值 {s['mean']:.4f} m, 最大 {s['max']:.4f} m")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(args.out, index=False, float_format="%.6f", encoding="utf-8-sig")
        print(f"\n[导出] 匹配结果已保存: {args.out}（{len(m)} 行，{len(m.columns)} 列）")
#（注：内容由AI生成）
