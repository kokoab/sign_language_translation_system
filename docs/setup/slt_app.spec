# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

# Paths
ROOT = os.path.abspath('.')
SRC = os.path.join(ROOT, 'src')
SCRIPTS = os.path.join(ROOT, 'scripts')

a = Analysis(
    [os.path.join(SRC, 'demo_classify.py')],
    pathex=[SRC, SCRIPTS, ROOT],
    binaries=[],
    datas=[
        # Model checkpoints
        ('models/output_v15_clean/best_model.pth', 'models/output_v15_clean'),
        ('models/output_stage2_v15_reextracted/stage2_best_model.pth', 'models/output_stage2_v15_reextracted'),
        # T5 model
        ('weights/slt_final_t5_model/model.safetensors', 'weights/slt_final_t5_model'),
        ('weights/slt_final_t5_model/config.json', 'weights/slt_final_t5_model'),
        ('weights/slt_final_t5_model/generation_config.json', 'weights/slt_final_t5_model'),
        ('weights/slt_final_t5_model/tokenizer.json', 'weights/slt_final_t5_model'),
        ('weights/slt_final_t5_model/tokenizer_config.json', 'weights/slt_final_t5_model'),
        ('weights/slt_final_t5_model/training_args.bin', 'weights/slt_final_t5_model'),
        # Manifest
        ('models/manifest.json', 'models'),
        # Source modules needed at runtime
        ('src/train_stage_1.py', '.'),
        ('src/train_stage_2.py', '.'),
        ('src/model_v14.py', '.'),
        ('src/model_v12.py', '.'),
        ('src/model_v11.py', '.'),
        ('src/camera_inference.py', '.'),
        ('src/extract.py', '.'),
        ('extract_batch_fast_v2.py', '.'),
        ('scripts/extract_apple_vision.py', '.'),
    ],
    hiddenimports=[
        'torch', 'torch.nn', 'torch.nn.functional',
        'torchvision',
        'transformers', 'transformers.models.t5',
        'tokenizers',
        'customtkinter',
        'cv2',
        'numpy',
        'PIL', 'PIL.Image', 'PIL.ImageTk',
        'scipy', 'scipy.signal',
        'safetensors', 'safetensors.torch',
        'huggingface_hub',
        'Vision', 'Quartz', 'Foundation', 'objc',
        'train_stage_1', 'train_stage_2',
        'model_v14', 'model_v12', 'model_v11',
        'extract_apple_vision', 'extract_batch_fast_v2', 'extract',
        'camera_inference',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter.test', 'nltk', 'yt_dlp'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ATLAS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ATLAS',
)

app = BUNDLE(
    coll,
    name='ATLAS.app',
    icon=None,
    bundle_identifier='com.atlas.asl',
    info_plist={
        'NSCameraUsageDescription': 'SLT needs camera access for sign language recognition.',
        'CFBundleDisplayName': 'ATLAS — ASL Translation and Language Assistive System',
        'CFBundleShortVersionString': '1.0.0',
        'NSHumanReadableCopyright': 'Francis Angelo Batiancela',
        'LSMultipleInstancesProhibited': True,
        'NSSupportsAutomaticTermination': False,
        'NSSupportsSuddenTermination': False,
    },
)
