# -*- coding: utf-8 -*-
"""
ecef_to_enu.py —— ECEF 大地直角坐标 → ENU 站心坐标转换（任务③ · 截止 9.10）
============================================================
功能
  - 读取 read_rtk 输出的 RTK/PPP 轨迹（含 X/Y/Z_ECEF_m、Lat_deg、Lon_deg、H_Ell_m）
  - 以指定参考历元（默认首历元）为本地原点，WGS84 标准旋转矩阵把 ECEF 差分转 ENU
  - 统一时间列 t_unix / t_rel（time_base.py 唯一权威），与 IMU 时间同步体系一致
  - 输出 CSV：t_unix, t_rel, E_m, N_m, U_m + 原始 ECEF/经纬高（供溯源）

坐标转换（WGS84 站心系）
  R = [[-sin(λ0),        cos(λ0),        0      ],
       [-sin(φ0)cos(λ0), -sin(φ0)sin(λ0), cos(φ0)],
       [ cos(φ0)cos(λ0),  cos(φ0)sin(λ0), sin(φ0)]]
  [E, N, U]^T = R · ([X, Y, Z]^T - [X0, Y0, Z0]^T)
  (φ0, λ0) 为参考点的 WGS84 椭球经纬度，(X0,Y0,Z0) 为其 ECEF 坐标。
  E=东、N=北、U=天顶；ENU 为当地水平坐标系，适合后续轨迹/位置分类的 2D 特征。

用法
  from read_rtk import read_rtk
  from ecef_to_enu import ecef_to_enu
  rtk = read_rtk('25号早上-329b.txt', leap_seconds=18)
  enu = ecef_to_enu(rtk)

命令行
  python ecef_to_enu.py 25号早上-329b.txt --out enu_329b.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from read_rtk import read_rtk
from time_base import rtk_to_unix, T_UNIX, T_REL

# WGS84 椭球参数
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2

# 输出列顺序
ENU_COLUMNS = [T_UNIX, T_REL, "E_m", "N_m", "U_m"]


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    """WGS84 大地坐标 (φ, λ, h) → ECEF (X, Y, Z)。"""
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(phi) ** 2)
    x = (n + h_m) * np.cos(phi) * np.cos(lam)
    y = (n + h_m) * np.cos(phi) * np.sin(lam)
    z = (n * (1.0 - WGS84_E2) + h_m) * np.sin(phi)
    return np.array([x, y, z], dtype=float)


def _ref_geodetic(rtk: pd.DataFrame, ref_index: int = 0) -> tuple[float, float, float, np.ndarray]:
    """取参考历元的 (φ0, λ0, h0)（十进制度）与其 ECEF 坐标 (X0,Y0,Z0)。"""
    r = rtk.iloc[ref_index]
    lat0, lon0, h0 = float(r["Lat_deg"]), float(r["Lon_deg"]), float(r["H_Ell_m"])
    xyz0 = rtk[["X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m"]].to_numpy(float)[ref_index]
    return lat0, lon0, h0, xyz0


def rotation_matrix_ecef_to_enu(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """WGS84 参考点 (φ0, λ0) 处的 ECEF→ENU 旋转矩阵（3x3，行序 E/N/U）。"""
    phi = np.deg2rad(lat0_deg)
    lam = np.deg2rad(lon0_deg)
    sp, cp = np.sin(phi), np.cos(phi)
    sl, cl = np.sin(lam), np.cos(lam)
    return np.array([
        [-sl,      cl,      0.0],
        [-sp * cl, -sp * sl, cp],
        [ cp * cl,  cp * sl, sp],
    ])


def ecef_to_enu(rtk: pd.DataFrame, ref_index: int = 0,
                ref_lat_deg: float | None = None,
                ref_lon_deg: float | None = None,
                ref_h_m: float | None = None,
                leap_seconds: float = 18.0) -> pd.DataFrame:
    """ECEF 轨迹 → ENU 站心坐标。

    :param rtk: read_rtk 输出的 DataFrame（需含 X/Y/Z_ECEF_m、Lat_deg、Lon_deg）
    :param ref_index: 本地原点取第几个历元（默认 0=首历元）；指定 ref_lat/ref_lon 后忽略
    :param ref_lat_deg: 手动指定参考点纬度（十进制度），如楼顶基准站坐标
    :param ref_lon_deg: 手动指定参考点经度（十进制度）
    :param ref_h_m: 手动指定参考点高程（m），默认 0
    :param leap_seconds: 与 time_base 一致，默认 18（GPS-18=UTC）
    :return: DataFrame，列为 t_unix, t_rel, E_m, N_m, U_m + 原始 ECEF/经纬高
    """
    if not {"X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m", "Lat_deg", "Lon_deg"} <= set(rtk.columns):
        raise ValueError("rtk 缺少 ECEF/经纬度列，请先经 read_rtk 读取")

    # 统一时间列（time_base 权威）
    df = rtk_to_unix(rtk, leap_seconds=leap_seconds)

    # 参考点与旋转矩阵
    if ref_lat_deg is not None and ref_lon_deg is not None:
        lat0, lon0 = float(ref_lat_deg), float(ref_lon_deg)
        h0 = float(ref_h_m) if ref_h_m is not None else 0.0
        xyz0 = geodetic_to_ecef(lat0, lon0, h0)
        ref_source = "手动指定"
    else:
        lat0, lon0, h0, xyz0 = _ref_geodetic(df, ref_index)
        ref_source = f"第{ref_index}历元"
    R = rotation_matrix_ecef_to_enu(lat0, lon0)

    # 矢量差分 + 旋转（每行一个 3 向量，按行乘 R^T 即列乘 R）
    dxyz = df[["X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m"]].to_numpy(float) - xyz0
    enu = dxyz @ R.T  # 行向量 [E, N, U]

    out = df[[T_UNIX, T_REL]].copy()
    out["E_m"], out["N_m"], out["U_m"] = enu[:, 0], enu[:, 1], enu[:, 2]
    # 溯源列：原始 ECEF + 经纬高 + 速度 + 参考点
    for c in ["X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m", "Lat_deg", "Lon_deg", "H_Ell_m",
              "VX_ECEF_mps", "VY_ECEF_mps", "VZ_ECEF_mps"]:
        out[c] = df[c].values
    out.attrs["ref_lat_deg"] = lat0
    out.attrs["ref_lon_deg"] = lon0
    out.attrs["ref_h_m"] = h0
    out.attrs["ref_index"] = ref_index
    out.attrs["ref_source"] = ref_source
    return out


def verify_enu(enu: pd.DataFrame) -> dict:
    """自检：参考点应为 (0,0,0)；ENU 相邻历元位移应等于 ECEF 位移；速度量级自洽。"""
    checks: dict = {}
    # 1) 参考点为零点
    e0, n0, u0 = enu.iloc[0][["E_m", "N_m", "U_m"]]
    checks["参考点E/N/U"] = f"({e0:.6f}, {n0:.6f}, {u0:.6f})"
    # 2) 相邻历元：ENU 3D 位移 vs ECEF 3D 位移（应一致）
    dxyz = np.diff(enu[["X_ECEF_m", "Y_ECEF_m", "Z_ECEF_m"]].to_numpy(float), axis=0)
    denu = np.diff(enu[["E_m", "N_m", "U_m"]].to_numpy(float), axis=0)
    dist_ecef = np.linalg.norm(dxyz, axis=1)
    dist_enu = np.linalg.norm(denu, axis=1)
    diff = np.abs(dist_ecef - dist_enu)
    checks["位移一致性"] = (f"相邻历元 ECEF/ENU 3D 位移差: 最大 {diff.max():.6e} m "
                          f"(应为 ~1e-9 量级)")
    # 3) 速度量级：水平位移 / dt 与文件 VX/VY/VZ 模长对比
    dt = np.diff(enu[T_REL].to_numpy(float))
    speed_enu = dist_ecef / np.where(dt == 0, np.nan, dt)  # 3D 速度
    v_file = np.linalg.norm(
        enu[["VX_ECEF_mps", "VY_ECEF_mps", "VZ_ECEF_mps"]].to_numpy(float), axis=1)
    ratio = speed_enu / np.where(v_file[1:] == 0, np.nan, v_file[1:])
    checks["速度自洽"] = (f"ENU 速度/文件速度 中位数 = {np.nanmedian(ratio):.3f} "
                         f"(应≈1.0)")
    # 4) 平面范围（轨迹尺寸）
    checks["轨迹平面范围"] = (f"E: {enu['E_m'].min():.2f}~{enu['E_m'].max():.2f} m, "
                            f"N: {enu['N_m'].min():.2f}~{enu['N_m'].max():.2f} m")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="ECEF → ENU 站心坐标（任务③）")
    parser.add_argument("file", help="IE 导出的 RTK 历元文件路径")
    parser.add_argument("--leap", type=float, default=18.0,
                        help="闰秒数（默认 18，得到 UTC 与 time_base 一致）")
    parser.add_argument("--ref-index", type=int, default=0,
                        help="本地原点取第几个历元（默认 0=首历元）；指定 --ref-lat 后忽略")
    parser.add_argument("--ref-lat", type=float, default=None,
                        help="手动指定参考点纬度（十进制度），如楼顶基准站 34.218582")
    parser.add_argument("--ref-lon", type=float, default=None,
                        help="手动指定参考点经度（十进制度），如楼顶基准站 117.144982")
    parser.add_argument("--ref-h", type=float, default=None,
                        help="手动指定参考点高程（m），默认 0")
    parser.add_argument("--out", default=None,
                        help="输出 CSV 路径（默认打印，不落盘）")
    args = parser.parse_args()

    rtk = read_rtk(args.file, leap_seconds=args.leap)
    enu = ecef_to_enu(rtk, ref_index=args.ref_index,
                       ref_lat_deg=args.ref_lat, ref_lon_deg=args.ref_lon, ref_h_m=args.ref_h,
                       leap_seconds=args.leap)

    lat0 = enu.attrs["ref_lat_deg"]
    lon0 = enu.attrs["ref_lon_deg"]
    h0 = enu.attrs["ref_h_m"]
    src = enu.attrs["ref_source"]
    print(f"历元数: {len(enu)}   参考点={src} (φ={lat0:.8f}°, λ={lon0:.8f}°, h={h0:.3f}m)")
    print(f"时间范围: t_rel {enu[T_REL].iloc[0]:.1f} ~ {enu[T_REL].iloc[-1]:.1f} s "
          f"({(enu[T_REL].iloc[-1] - enu[T_REL].iloc[0]) / 60:.1f} min)")
    print("\n===== ENU 自检 =====")
    for k, v in verify_enu(enu).items():
        print(f"  {k:<16}: {v}")
    print("\n前 5 行（t_unix, t_rel, E, N, U）:")
    print(enu[ENU_COLUMNS].head().to_string(index=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        enu.to_csv(args.out, index=False, float_format="%.6f", encoding="utf-8-sig")
        print(f"\n已保存: {args.out}")


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
