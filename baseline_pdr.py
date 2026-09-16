"""
baseline_pdr.py
传统PDR基线核心函数

包含：
1. step_detection()              —— 峰值法步态检测
2. step_length_weinberg()        —— Weinberg步长估计模型
3. heading_complementary_filter() —— 互补滤波航向估计
4. position_update()             —— 位置递推

日期：2026.09
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# 1. 步态检测（峰值法）
# ============================================================

def lowpass_filter(data, fs, cutoff=5.0, order=4):
    """
    低通滤波器（Butterworth）

    参数：
        data: 输入信号（1D数组）
        fs: 采样频率（Hz）
        cutoff: 截止频率（Hz），默认5Hz
        order: 滤波器阶数，默认4

    返回：
        滤波后的信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def step_detection(acc_data, fs=100, peak_height=2.0, peak_distance=20, peak_prominence=1.0):
    """
    峰值法步态检测

    原理：通过合成加速度的峰值检测识别迈步事件

    参数：
        acc_data: dict 或 DataFrame，包含 acc_x, acc_y, acc_z
        fs: 采样频率（Hz），默认100
        peak_height: 峰值最小高度（m/s^2），默认2.0
        peak_distance: 峰值最小间距（采样点数），默认20
        peak_prominence: 峰值显著性，默认1.0

    返回：
        step_indices: 步态事件对应的采样点索引（numpy数组）
        step_intervals: 相邻步之间的采样点数（numpy数组）
    """
    # 提取三轴加速度
    if isinstance(acc_data, dict):
        acc_x = np.array(acc_data['acc_x'])
        acc_y = np.array(acc_data['acc_y'])
        acc_z = np.array(acc_data['acc_z'])
    else:
        acc_x = acc_data['acc_x'].values
        acc_y = acc_data['acc_y'].values
        acc_z = acc_data['acc_z'].values

    # 计算合成加速度
    acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

    # 低通滤波（去除高频噪声）
    acc_mag_filtered = lowpass_filter(acc_mag, fs, cutoff=5.0)

    # 去除重力影响（减去均值）
    acc_mag_centered = acc_mag_filtered - np.mean(acc_mag_filtered)

    # 峰值检测
    peaks, properties = find_peaks(
        acc_mag_centered,
        height=peak_height,
        distance=peak_distance,
        prominence=peak_prominence
    )

    # 计算步间间隔
    if len(peaks) > 1:
        step_intervals = np.diff(peaks)
    else:
        step_intervals = np.array([])

    return peaks, step_intervals


# ============================================================
# 2. 步长估计（Weinberg模型）
# ============================================================

def step_length_weinberg(acc_data, step_indices, fs=100, K=0.45):
    """
    Weinberg步长估计模型

    公式：SL = K × (A_max - A_min)^(1/4)
    其中 A_max 和 A_min 是单步周期内合成加速度的最大值和最小值

    参数：
        acc_data: dict 或 DataFrame，包含 acc_x, acc_y, acc_z
        step_indices: 步态事件对应的采样点索引（来自step_detection）
        fs: 采样频率（Hz），默认100
        K: 个性化校准系数，默认0.45（需根据测试者调整）

    返回：
        step_lengths: 每一步的步长估计值（米），numpy数组
    """
    # 提取三轴加速度
    if isinstance(acc_data, dict):
        acc_x = np.array(acc_data['acc_x'])
        acc_y = np.array(acc_data['acc_y'])
        acc_z = np.array(acc_data['acc_z'])
    else:
        acc_x = acc_data['acc_x'].values
        acc_y = acc_data['acc_y'].values
        acc_z = acc_data['acc_z'].values

    # 计算合成加速度
    acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    acc_mag_filtered = lowpass_filter(acc_mag, fs, cutoff=5.0)

    step_lengths = []

    for i in range(len(step_indices) - 1):
        start = step_indices[i]
        end = step_indices[i + 1]

        # 单步周期内的加速度段
        acc_segment = acc_mag_filtered[start:end]

        if len(acc_segment) < 2:
            continue

        A_max = np.max(acc_segment)
        A_min = np.min(acc_segment)

        # Weinberg公式
        if A_max > A_min:
            SL = K * (A_max - A_min) ** 0.25
        else:
            SL = 0.5  # 异常情况给默认值

        step_lengths.append(SL)

    return np.array(step_lengths)


# ============================================================
# 3. 航向估计（互补滤波）
# ============================================================

def heading_complementary_filter(gyro_data, ahrs_data, fs=100, alpha=0.98,
                                 remove_bias=True):
    """
    互补滤波航向估计

    原理：陀螺仪积分得到短期航向变化，磁力计（Yaw）提供长期绝对参考。
          互补滤波融合两者：
              heading[i] = alpha * (heading[i-1] + gyro_z * dt)
                         + (1 - alpha) * yaw_mag
          其中 alpha 越大，越信任陀螺仪（短期平滑但会漂移）；
          alpha 越小，越信任磁力计（长期不漂但噪声大）。

    改进（相对原始版）：
      1) 融合前对 yaw 做 np.unwrap，避免 ±180° 处假跳变；
      2) 融合时把 (yaw - gyro_pred) 归一到 -π~π，避免绕圈；
      3) 可选扣除 gyro_z 的均值（零偏），抑制航向长期漂移。

    参数：
        gyro_data: dict 或 DataFrame，包含 gyro_x, gyro_y, gyro_z
        ahrs_data: dict 或 DataFrame，包含 yaw（磁力计航向，单位：度）
        fs: 采样频率（Hz），默认100
        alpha: 互补滤波系数（陀螺仪权重），默认0.98；
               陀螺漂移明显时建议降到 0.85~0.95
        remove_bias: 是否扣除 gyro_z 的均值（零偏），默认 True

    返回：
        headings: 每一步的航向估计值（弧度，连续角，可能超出 ±π），numpy数组
    """
    # 提取陀螺仪Z轴角速度
    if isinstance(gyro_data, dict):
        gyro_z = np.array(gyro_data['gyro_z'])
    else:
        gyro_z = gyro_data['gyro_z'].values

    # 提取磁力计航向（Yaw，度）
    if isinstance(ahrs_data, dict):
        yaw_mag = np.array(ahrs_data['yaw'])
    else:
        yaw_mag = ahrs_data['yaw'].values

    # 确保长度一致
    min_len = min(len(gyro_z), len(yaw_mag))
    gyro_z = gyro_z[:min_len].astype(float)
    yaw_mag = yaw_mag[:min_len].astype(float)

    # 扣除陀螺仪零偏（抑制航向长期漂移）
    if remove_bias:
        gyro_z = gyro_z - np.mean(gyro_z)

    # Yaw: 度 -> 弧度，并解缠成连续角（避免 ±180° 处假跳变）
    yaw_mag_rad = np.radians(yaw_mag)
    yaw_mag_rad = np.unwrap(yaw_mag_rad)

    dt = 1.0 / fs
    headings = np.zeros(min_len)
    headings[0] = yaw_mag_rad[0]

    for i in range(1, min_len):
        # 陀螺仪积分预测
        gyro_pred = headings[i - 1] + gyro_z[i] * dt
        # 融合前把差值归一到 -π~π
        diff = yaw_mag_rad[i] - gyro_pred
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        # 互补融合
        headings[i] = gyro_pred + (1 - alpha) * diff

    return headings


# ============================================================
# 4. 位置递推（辅助函数）
# ============================================================

def position_update(step_indices, step_lengths, headings, use_midpoint=True):
    """
    根据步态、步长、航向递推位置

    参数：
        step_indices: 步态事件对应的采样点索引（长度 = n_steps + 1）
        step_lengths: 每一步的步长（米，长度 = n_steps）
        headings:     航向序列（弧度，长度 = 采样点数）
        use_midpoint: True 取步中点的航向；False 取步终点的航向（旧行为）

    返回：
        positions: 位置序列（(n_steps+1) × 2 数组），每行为 (东, 北)
    """
    n_steps = min(len(step_lengths), len(headings) - 1)
    positions = np.zeros((n_steps + 1, 2))

    for i in range(n_steps):
        # 取这一步的航向代表值
        if use_midpoint and i + 1 < len(step_indices):
            idx = int((step_indices[i] + step_indices[i + 1]) // 2)  # 步中点
        else:
            idx = step_indices[i + 1] if i + 1 < len(step_indices) else step_indices[-1]
        heading = headings[idx] if idx < len(headings) else headings[-1]

        # 位置递推（x=东，y=北；heading 为连续角，sin/cos 同样适用）
        positions[i + 1, 0] = positions[i, 0] + step_lengths[i] * np.sin(heading)
        positions[i + 1, 1] = positions[i, 1] + step_lengths[i] * np.cos(heading)

    return positions


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    # 生成模拟数据测试
    fs = 100
    t = np.arange(0, 10, 1 / fs)

    # 模拟加速度（含周期性步态信号）
    acc_x = 2.0 * np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))
    acc_y = 1.5 * np.sin(2 * np.pi * 2 * t + 0.5) + 0.1 * np.random.randn(len(t))
    acc_z = 9.8 + 3.0 * np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))

    acc_data = {'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z}

    # 步态检测
    peaks, intervals = step_detection(acc_data, fs=fs)
    print(f"检测到步数: {len(peaks)}")
    print(f"步间间隔（采样点）: {intervals[:5]}...")

    # 步长估计
    step_lengths = step_length_weinberg(acc_data, peaks, fs=fs)
    print(f"步长估计（前5步）: {step_lengths[:5]}")

    # 模拟陀螺仪和AHRS数据
    gyro_z = 0.5 * np.sin(2 * np.pi * 0.5 * t) + 0.01 * np.random.randn(len(t))
    yaw_mag = 30 * np.sin(2 * np.pi * 0.1 * t)

    gyro_data = {'gyro_z': gyro_z}
    ahrs_data = {'yaw': yaw_mag}

    # 航向估计
    headings = heading_complementary_filter(gyro_data, ahrs_data, fs=fs)
    print(f"航向估计（前5个）: {headings[:5]}")

    # 位置递推
    positions = position_update(peaks, step_lengths, headings)
    print(f"位置序列形状: {positions.shape}")
    print(f"终点位置: {positions[-1]}")