#!/usr/bin/env python3
"""Monitor script to run training and collect resource usage.

Usage examples:
  python3 tools/train_monitor.py --days 1 --poll 1 --preserve-model
  python3 tools/train_monitor.py --all --poll 2

The script will run: python3 -m src.collector.train_model [--days N | --all]
and sample CPU, memory and IO while the process runs. It writes a CSV with samples
and a JSON summary at the end. If --preserve-model is set, the produced model.joblib
is renamed to model_YYYYmmdd_HHMMSS.joblib to avoid overwriting previous models.
"""

import argparse
import subprocess
import time
import os
import sys
import csv
import json
from datetime import datetime

def has_psutil():
    try:
        import psutil
        return True
    except Exception:
        return False

def read_proc_status(pid):
    # returns rss_kb or None
    try:
        with open(f"/proc/{pid}/status","r") as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    return int(parts[1]) # kB
    except Exception:
        return None

def read_proc_io(pid):
    # returns (read_bytes, write_bytes) or (None, None)
    try:
        rd = None; wr = None
        with open(f"/proc/{pid}/io","r") as f:
            for line in f:
                if line.startswith('read_bytes:'):
                    rd = int(line.split()[1])
                if line.startswith('write_bytes:'):
                    wr = int(line.split()[1])
        return rd, wr
    except Exception:
        return None, None

def read_ps_cpu(pid):
    try:
        out = subprocess.check_output(['ps','-p',str(pid),'-o','%cpu=']).decode().strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0

def monitor_process(proc, poll_interval=1.0):
    use_psutil = has_psutil()
    if use_psutil:
        import psutil
        p = psutil.Process(proc.pid)
        # Primeira chamada para inicializar cpu_percent
        try:
            p.cpu_percent(interval=None)
        except Exception:
            pass

    samples = []
    start_ts = time.time()
    first_io = (None, None)
    last_io = (None, None)
    try:
        while True:
            if proc.poll() is not None:
                break
            ts = time.time()
            if use_psutil:
                try:
                    cpu = p.cpu_percent(interval=None)
                    mem = p.memory_info().rss  # bytes
                    io = p.io_counters() if hasattr(p, 'io_counters') else None
                    read_b = io.read_bytes if io is not None else None
                    write_b = io.write_bytes if io is not None else None
                except Exception:
                    cpu = read_ps_cpu(proc.pid)
                    rss_kb = read_proc_status(proc.pid)
                    mem = rss_kb * 1024 if rss_kb is not None else None
                    read_b, write_b = read_proc_io(proc.pid)
            else:
                cpu = read_ps_cpu(proc.pid)
                rss_kb = read_proc_status(proc.pid)
                mem = rss_kb * 1024 if rss_kb is not None else None
                read_b, write_b = read_proc_io(proc.pid)

            if first_io == (None, None) and (read_b is not None or write_b is not None):
                first_io = (read_b or 0, write_b or 0)
            last_io = (read_b or 0, write_b or 0)

            samples.append({'ts': ts, 'cpu_percent': cpu, 'mem_bytes': mem, 'read_bytes': read_b, 'write_bytes': write_b})
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print('Interrupted by user, terminating monitor...')
    end_ts = time.time()
    return samples, start_ts, end_ts, first_io, last_io

def summarize(samples, start_ts, end_ts, first_io, last_io, model_path=None):
    import statistics
    mem_peaks = [s['mem_bytes'] for s in samples if s['mem_bytes'] is not None]
    cpu_samples = [s['cpu_percent'] for s in samples if s['cpu_percent'] is not None]
    read_samples = [s['read_bytes'] for s in samples if s['read_bytes'] is not None]
    write_samples = [s['write_bytes'] for s in samples if s['write_bytes'] is not None]

    summary = {
        'start_time': datetime.fromtimestamp(start_ts).isoformat(),
        'end_time': datetime.fromtimestamp(end_ts).isoformat(),
        'duration_seconds': round(end_ts - start_ts, 2),
        'samples': len(samples),
        'cpu_avg_percent': round(statistics.mean(cpu_samples), 2) if cpu_samples else None,
        'cpu_max_percent': round(max(cpu_samples), 2) if cpu_samples else None,
        'mem_peak_bytes': int(max(mem_peaks)) if mem_peaks else None,
        'mem_peak_mb': round(max(mem_peaks)/1024/1024, 2) if mem_peaks else None,
        'total_read_bytes': (last_io[0] - first_io[0]) if (first_io[0] is not None and last_io[0] is not None) else None,
        'total_write_bytes': (last_io[1] - first_io[1]) if (first_io[1] is not None and last_io[1] is not None) else None,
        'sample_interval_seconds': round((end_ts - start_ts)/len(samples), 2) if samples else None
    }
    if model_path and os.path.exists(model_path):
        summary['model_path'] = model_path
        summary['model_size_bytes'] = os.path.getsize(model_path)
        summary['model_size_mb'] = round(summary['model_size_bytes']/1024/1024, 2)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=90, help='window days to pass to train script (default 90). Use 0 or negative for all data via --all')
    parser.add_argument('--all', action='store_true', help='pass --all to train script (use all data)')
    parser.add_argument('--poll', type=float, default=1.0, help='monitor polling interval seconds')
    parser.add_argument('--out-dir', type=str, default='train_monitor_output', help='directory to store logs and reports')
    parser.add_argument('--preserve-model', action='store_true', help='after training finishes, rename model.joblib to include timestamp to preserve it')
    parser.add_argument('--n-estimators', type=int, default=None, help='forwarded to train script: n_estimators for RandomForest')
    parser.add_argument('--n-jobs', type=int, default=None, help='forwarded to train script: n_jobs for RandomForest')
    parser.add_argument('--max-features', type=str, default=None, help='forwarded to train script: max_features for RandomForest (sqrt, log2, None)')
    parser.add_argument('--baseline', action='store_true', help='forwarded to train script: use only original 4 features for baseline comparison')
    parser.add_argument('--temporal-split', action='store_true', help='forwarded to train script: use temporal train/test split instead of random')
    parser.add_argument('--clean-data', action='store_true', help='forwarded to train script: use cleaned data without outliers')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.out_dir, f'train_log_{ts}.txt')
    csv_file = os.path.join(args.out_dir, f'monitor_samples_{ts}.csv')
    report_file = os.path.join(args.out_dir, f'train_report_{ts}.json')

    cmd = [sys.executable, '-m', 'src.collector.train_model']
    if args.all:
        cmd.append('--all')
    else:
        cmd += ['--days', str(args.days)]
    # forward optional RF params
    if args.n_estimators is not None:
        cmd += ['--n-estimators', str(args.n_estimators)]
    if args.n_jobs is not None:
        cmd += ['--n-jobs', str(args.n_jobs)]
    if args.max_features is not None:
        cmd += ['--max-features', str(args.max_features)]
    if args.baseline:
        cmd.append('--baseline')
    if args.clean_data:
        cmd.append('--clean-data')
    if args.temporal_split:
        cmd.append('--temporal-split')

    print('Running training command:', ' '.join(cmd))
    print('Logs will be saved to', log_file)

    with open(log_file, 'wb') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

        samples, start_ts, end_ts, first_io, last_io = monitor_process(proc, poll_interval=args.poll)

        # Ensure process finished
        ret = proc.wait()
        print('Training process exited with code', ret)

    # write samples CSV
    with open(csv_file, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['ts','cpu_percent','mem_bytes','read_bytes','write_bytes'])
        writer.writeheader()
        for s in samples:
            writer.writerow(s)

    # Optionally preserve model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, 'model.joblib')
    preserved_path = None
    if args.preserve_model and os.path.exists(model_path):
        preserved_path = os.path.join(args.out_dir, f"model_{ts}.joblib")
        try:
            os.replace(model_path, preserved_path)
            print('Preserved model to', preserved_path)
        except Exception as e:
            print('Failed to preserve model:', e)

    summary = summarize(samples, start_ts, end_ts, first_io, last_io, model_path=preserved_path or (model_path if os.path.exists(model_path) else None))
    with open(report_file, 'w') as rf:
        json.dump(summary, rf, indent=2)

    print('\nSummary:')
    print(json.dumps(summary, indent=2))
    print('\nSamples saved to', csv_file)
    print('Full training log at', log_file)
    print('Report saved to', report_file)

if __name__ == '__main__':
    main()
