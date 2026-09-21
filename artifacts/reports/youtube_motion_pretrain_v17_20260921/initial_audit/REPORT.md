# Pilot execution failure

Training did not start.

```
Traceback (most recent call last):
  File "/Users/frnzlo/Documents/machine_learning/SLT/scripts/run_youtube_motion_pilot_v17.py", line 70, in main
    subprocess.run([sys.executable, "scripts/audit_youtube_motion_pilot_v17.py",
  File "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/subprocess.py", line 528, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/Users/frnzlo/Documents/machine_learning/SLT/venv/bin/python', 'scripts/audit_youtube_motion_pilot_v17.py', '--manifest', 'artifacts/reports/free_continuous_asl_alternatives_20260921/acquired_manifest.csv', '--input-root', 'data/local/youtube_asl_keypoints_20260921/pilot_2000_raw', '--output-dir', '/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/youtube_motion_pretrain_v17_20260921']' returned non-zero exit status 1.
```
