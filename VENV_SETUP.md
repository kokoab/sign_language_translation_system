# Virtual Environment Setup

Use a fresh venv to avoid conflicts and the `mediapipe has no attribute 'solutions'` error.

## 1. Create the venv

**Windows (Command Prompt or PowerShell):**
```cmd
cd C:\path\to\SLT
python -m venv venv
venv\Scripts\activate
```

**Windows (Git Bash):**
```bash
cd /c/path/to/SLT
python -m venv venv
source venv/Scripts/activate
```

**macOS / Linux:**
```bash
cd /path/to/SLT
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your prompt.

---

## 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verify MediaPipe

```bash
python -c "import mediapipe as mp; print(mp.solutions.hands); print('OK')"
```

If you see `OK`, extraction will work.

---

## If you still get "module 'mediapipe' has no attribute 'solutions'"

1. **Check 64-bit Python** (MediaPipe needs 64-bit on Windows):
   ```bash
   python -c "import struct; print(struct.calcsize('P')*8, 'bit')"
   ```
   Must show `64 bit`. If it shows `32 bit`, install 64-bit Python.

2. **No local shadow:** Ensure there is no folder or file named `mediapipe` in your project directory. Rename it if present.

3. **Clean reinstall:**
   ```bash
   pip uninstall mediapipe -y
   pip install mediapipe==0.10.10 --no-cache-dir
   ```

4. **Python version:** Use Python 3.8–3.11. MediaPipe may not work on 3.12+ on some platforms.
