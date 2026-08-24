"""
Build SLT Sign Language Translator as a Mac .app bundle.

Usage:
    python setup_app.py py2app
"""
from setuptools import setup
import os

# All data files to bundle
DATA_FILES = [
    ('models/output_v15_clean', [
        'models/output_v15_clean/best_model.pth',
    ]),
    ('models/output_stage2_v15_reextracted', [
        'models/output_stage2_v15_reextracted/stage2_best_model.pth',
    ]),
    ('weights/slt_final_t5_model', [
        'weights/slt_final_t5_model/model.safetensors',
        'weights/slt_final_t5_model/config.json',
        'weights/slt_final_t5_model/generation_config.json',
        'weights/slt_final_t5_model/tokenizer.json',
        'weights/slt_final_t5_model/tokenizer_config.json',
        'weights/slt_final_t5_model/training_args.bin',
    ]),
    ('models', [
        'models/manifest.json',
    ]),
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': None,
    'plist': {
        'CFBundleName': 'SLT Translator',
        'CFBundleDisplayName': 'SLT — Sign Language Translator',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSCameraUsageDescription': 'SLT needs camera access for sign language recognition.',
        'NSHumanReadableCopyright': 'Francis Angelo Batiancela — Leyte Normal University',
    },
    'packages': [
        'torch', 'torchvision', 'transformers', 'tokenizers',
        'customtkinter', 'cv2', 'numpy', 'PIL',
        'Vision', 'Quartz', 'Foundation', 'objc',
        'scipy', 'safetensors', 'huggingface_hub',
    ],
    'includes': [
        'train_stage_1', 'train_stage_2', 'model_v14', 'model_v12', 'model_v11',
        'extract_apple_vision', 'extract_batch_fast_v2', 'extract',
        'camera_inference',
    ],
    'excludes': [
        'matplotlib', 'tkinter.test', 'test', 'unittest',
        'email', 'html', 'http', 'xmlrpc', 'pydoc',
    ],
    'resources': [],
    'site_packages': True,
}

setup(
    name='SLT Translator',
    app=['src/demo_classify.py'],
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
