"""Sustained concurrent load on the phone: hand landmarks (GPU) + span recognizer B8 (4 big cores)."""
import re, subprocess, json, sys
ADB = 'artifacts/generated/android_tools/platform-tools/adb'
def sh(c): return subprocess.run([ADB, 'shell', c], capture_output=True, text=True, timeout=300).stdout
def temp():
    m = re.search(r'temperature: (\d+)', sh('dumpsys battery')); return int(m[1]) / 10 if m else None
def avg(log):
    m = re.search(r'Inference \(avg\): ([\d.e+]+)', log); return round(float(m[1]) / 1000, 1) if m else None
rows = []
for i in range(1, int(sys.argv[1]) + 1):
    t0 = temp()
    sh('cd /data/local/tmp/slt_bench && (taskset f0 ./benchmark_model --graph=hand_landmarkerhand_landmarks_detector.tflite '
       '--use_gpu=true --num_runs=100000 --min_secs=30 --max_secs=30 --warmup_runs=1 > g.log 2>&1 &); '
       'taskset f0 ./benchmark_model --graph=span_recognizer_mediapipe_b8_fp32.tflite --num_threads=4 --num_runs=100000 '
       '--min_secs=30 --max_secs=30 --warmup_runs=1 > c.log 2>&1; sleep 2')
    row = dict(chunk=i, battery_c_start=t0, battery_c_end=temp(), hand_gpu_ms=avg(sh('cat /data/local/tmp/slt_bench/g.log')),
               span_cpu_ms=avg(sh('cat /data/local/tmp/slt_bench/c.log')),
               big_core_khz=sh('cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq').strip(),
               gpu_freq=sh('cat /sys/class/devfreq/*gpu*/cur_freq 2>/dev/null | head -1').strip())
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open('artifacts/reports/android_device_bench_20261004/sustained_concurrent.json', 'w'), indent=1)
