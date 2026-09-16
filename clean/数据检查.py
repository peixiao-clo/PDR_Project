import os
import sys
import argparse
import csv
from pathlib import Path

# ================== 工具函数 ==================

def read_file_lines(filepath):
    """读取文件内容，返回头部注释行和数据行，并处理编码错误"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return None, None, str(e)

    header_lines = []
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('%'):
            header_lines.append(stripped)
        else:
            data_lines.append(stripped)
    return header_lines, data_lines, None

def extract_app_times(data_lines):
    """从数据行中提取所有 AppTimestamp（第二列），返回排序后的列表"""
    times = []
    for line in data_lines:
        parts = line.split(';')
        if len(parts) >= 2:
            try:
                t = float(parts[1])
                times.append(t)
            except ValueError:
                continue
    return sorted(times)

def find_posi_threshold(data_lines, counter_value):
    """查找指定 POSI Counter 值的 AppTimestamp，若未找到返回 None"""
    for line in data_lines:
        if line.startswith('POSI;'):
            parts = line.split(';')
            if len(parts) >= 3:
                try:
                    ts = float(parts[1])
                    cnt = int(parts[2])
                    if cnt == counter_value:
                        return ts
                except ValueError:
                    continue
    return None

def check_time_anomalies(times):
    """分析时间戳连续性，返回最大间隔、大间隔次数（>1秒）和是否有回跳"""
    if len(times) < 2:
        return 0.0, 0, False
    diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
    max_gap = max(diffs)
    large_gaps = sum(1 for d in diffs if d > 1.0)
    has_jump_back = any(d < 0 for d in diffs)
    return max_gap, large_gaps, has_jump_back

# ================== 完整性检查 ==================

def integrity_check(folder_path, output_report_path):
    """遍历文件夹，检查所有 .txt 文件，生成 CSV 报告"""
    # 获取所有 .txt 文件
    files = list(Path(folder_path).glob('*.txt'))
    if not files:
        print(f"[警告] 在 {folder_path} 中未找到任何 .txt 文件。")
        return False

    report_rows = []
    for file_path in files:
        fname = file_path.name
        size = file_path.stat().st_size
        header, data_lines, error = read_file_lines(file_path)

        # 基本信息
        damaged = (error is not None) or (data_lines is not None and len(data_lines) == 0)
        reason = error if error else ('OK' if not damaged else '文件为空')
        data_count = len(data_lines) if data_lines else 0

        # 传感器类型
        sensors = set()
        posi_counters = []
        app_times = []
        if data_lines:
            for line in data_lines:
                parts = line.split(';')
                if parts:
                    sensors.add(parts[0])
                    if parts[0] == 'POSI' and len(parts) >= 3:
                        try:
                            posi_counters.append(int(parts[2]))
                        except ValueError:
                            pass
                if len(parts) >= 2:
                    try:
                        app_times.append(float(parts[1]))
                    except ValueError:
                        pass

        has_header = bool(header)
        has_posi2 = 2 in posi_counters if not damaged else False
        has_posi3 = 3 in posi_counters if not damaged else False

        # 时间连续性
        sorted_times = sorted(app_times)
        time_span = (sorted_times[-1] - sorted_times[0]) if sorted_times else 0.0
        max_gap, large_gaps, jump_back = check_time_anomalies(sorted_times) if sorted_times else (0.0, 0, False)

        row = {
            '文件名': fname,
            '大小(bytes)': size,
            '损坏': damaged,
            '损坏原因': reason,
            '头部注释': has_header,
            '数据行数': data_count,
            '传感器类型': ','.join(sorted(sensors)),
            'POSI总数': len(posi_counters),
            '存在POSI_2': has_posi2,
            '存在POSI_3': has_posi3,
            '时间跨度(秒)': round(time_span, 3),
            '最大间隔(秒)': round(max_gap, 3),
            '大间隔次数(>1s)': large_gaps,
            '时间戳回跳': jump_back
        }
        report_rows.append(row)

    # 写入CSV
    if report_rows:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        fieldnames = list(report_rows[0].keys())
        with open(output_report_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"[✓] 完整性报告已保存至: {output_report_path}")

        # 简要屏幕输出
        damaged_files = [r for r in report_rows if r['损坏']]
        missing_posi2 = [r for r in report_rows if not r['存在POSI_2']]
        missing_posi3 = [r for r in report_rows if not r['存在POSI_3']]
        print(f"共检查 {len(report_rows)} 个文件。")
        if damaged_files:
            print(f"损坏/空文件: {len(damaged_files)} 个:")
            for r in damaged_files:
                print(f"  - {r['文件名']}: {r['损坏原因']}")
        if missing_posi2:
            print(f"缺少 POSI 2 的文件: {len(missing_posi2)} 个")
        if missing_posi3:
            print(f"缺少 POSI 3 的文件: {len(missing_posi3)} 个")
        return True
    else:
        print("未生成报告。")
        return False

# ================== 数据清洗 ==================

def clean_single_file(input_path, output_path):
    """清洗单个文件，截取 POSI 2 到 POSI 3 之间的数据，并返回统计信息"""
    header, data_lines, error = read_file_lines(input_path)
    if error:
        print(f"[错误] 无法读取 {input_path.name}: {error}")
        return None

    # 寻找 POSI 2 和 POSI 3 的时间戳
    t2 = find_posi_threshold(data_lines, 2)
    t3 = find_posi_threshold(data_lines, 3)

    if t2 is None or t3 is None:
        print(f"[警告] 文件 {input_path.name} 缺少 POSI 2 或 POSI 3，将使用全部数据作为行走段。")
        # 确定一个默认时间范围：若缺少一个端点，用整个数据的时间范围代替
        all_times = extract_app_times(data_lines)
        if not all_times:
            print(f"[错误] 文件 {input_path.name} 没有有效的时间戳，跳过清洗。")
            return None
        if t2 is None:
            t2 = all_times[0]
        if t3 is None:
            t3 = all_times[-1]

    # 筛选数据行 (POSI 2 <= AppTimestamp <= POSI 3)
    cleaned_lines = []
    for line in data_lines:
        parts = line.split(';')
        if len(parts) < 2:
            # 没有时间戳的行也保留，但可能为异常行
            cleaned_lines.append(line)
            continue
        try:
            ts = float(parts[1])
        except ValueError:
            cleaned_lines.append(line)
            continue
        if ts < t2 or ts > t3:
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        print(f"[错误] 清洗后无数据，文件: {input_path.name} (t2={t2}, t3={t3})")
        return None

    # 写入清洗后文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for hline in header:
            f.write(hline + '\n')
        for dline in cleaned_lines:
            f.write(dline + '\n')

    # 统计
    num_points = len(cleaned_lines)
    duration = t3 - t2

    # 异常检测：基于清洗后数据的时间戳
    cleaned_times = []
    for line in cleaned_lines:
        parts = line.split(';')
        if len(parts) >= 2:
            try:
                cleaned_times.append(float(parts[1]))
            except:
                pass
    cleaned_times.sort()
    max_gap, large_gaps, jump_back = check_time_anomalies(cleaned_times) if len(cleaned_times) > 1 else (0.0, 0, False)

    anomalies = []
    if large_gaps > 0:
        anomalies.append(f"大间隔次数: {large_gaps} (max {max_gap:.3f}s)")
    if jump_back:
        anomalies.append("时间戳存在回跳")
    if not cleaned_times:
        anomalies.append("无有效时间戳")

    stats = {
        'filename': input_path.name,
        'duration': round(duration, 3),
        'data_points': num_points,
        't2': t2,
        't3': t3,
        'anomalies': anomalies,
        'output_file': output_path.name
    }
    return stats

def batch_clean(folder_path, output_dir):
    """批量清洗 folder_path 下的所有 .txt 文件，返回汇总统计列表"""
    files = list(Path(folder_path).glob('*.txt'))
    if not files:
        print(f"[警告] 未找到 .txt 文件于 {folder_path}")
        return []

    all_stats = []
    for file_path in files:
        output_path = Path(output_dir) / (file_path.stem + '_cleaned' + file_path.suffix)
        stats = clean_single_file(file_path, output_path)
        if stats:
            all_stats.append(stats)
            # 屏幕输出单个文件清洗结果
            print(f"[✓] {stats['filename']} -> {stats['output_file']} | 时长={stats['duration']}s, 数据点={stats['data_points']}")
            if stats['anomalies']:
                print(f"     异常: {'; '.join(stats['anomalies'])}")
    return all_stats

def write_cleaning_summary(stats_list, summary_path):
    """将清洗统计列表写入文本文件"""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=========== 数据清洗汇总报告 ===========\n")
        f.write(f"处理文件数: {len(stats_list)}\n\n")
        for s in stats_list:
            f.write(f"文件: {s['filename']}\n")
            f.write(f"  行走时长: {s['duration']} 秒\n")
            f.write(f"  有效数据点数: {s['data_points']}\n")
            f.write(f"  POSI2 时间戳: {s['t2']}\n")
            f.write(f"  POSI3 时间戳: {s['t3']}\n")
            if s['anomalies']:
                f.write(f"  异常: {'; '.join(s['anomalies'])}\n")
            else:
                f.write(f"  数据连续性正常\n")
            f.write(f"  输出文件: {s['output_file']}\n\n")
    print(f"[✓] 清洗汇总报告已保存至: {summary_path}")

# ================== 主函数 ==================

def main():
    parser = argparse.ArgumentParser(
        description="IMU 数据完整性检查与行走轨迹清洗工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python process_imu_data.py                           # 默认处理 ./data 文件夹
  python process_imu_data.py --input_dir ./raw --output_dir ./out
  python process_imu_data.py --no_clean                 # 仅生成完整性报告
        """
    )
    # 默认路径：脚本所在目录下的 data 和 cleaned
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--input_dir', type=str, default=os.path.join(script_dir, 'data'),
                        help='原始数据文件夹路径 (默认: 脚本同目录下的 data 文件夹)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(script_dir, 'cleaned'),
                        help='输出文件夹路径 (默认: 脚本同目录下的 cleaned 文件夹)')
    parser.add_argument('--report_file', type=str, default='integrity_report.csv',
                        help='完整性报告文件名 (默认: integrity_report.csv)')
    parser.add_argument('--summary_file', type=str, default='cleaning_summary.txt',
                        help='清洗汇总文件名 (默认: cleaning_summary.txt)')
    parser.add_argument('--no_clean', action='store_true',
                        help='跳过数据清洗，只进行完整性检查')
    args = parser.parse_args()

    # 确保输入文件夹存在
    if not os.path.isdir(args.input_dir):
        print(f"[错误] 输入文件夹不存在: {args.input_dir}")
        sys.exit(1)

    # 1. 完整性检查
    report_path = os.path.join(args.output_dir, args.report_file)
    print("==================== 开始完整性检查 ====================")
    success = integrity_check(args.input_dir, report_path)
    if not success:
        print("完整性检查未生成有效报告，请检查输入数据。")
        if args.no_clean:
            sys.exit(0)

    # 2. 数据清洗
    if not args.no_clean:
        print("\n==================== 开始数据清洗 ====================")
        stats_list = batch_clean(args.input_dir, args.output_dir)
        if stats_list:
            summary_path = os.path.join(args.output_dir, args.summary_file)
            write_cleaning_summary(stats_list, summary_path)
        else:
            print("[警告] 没有文件被成功清洗。")
    else:
        print("\n已跳过数据清洗步骤。")

if __name__ == '__main__':
    main()