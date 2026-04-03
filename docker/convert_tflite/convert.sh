#!/bin/bash
set -e
echo "Converting RTMW-XL ONNX to TFLite..."
echo "Input: /work/rtmw.onnx"
echo "Output: /work/output/"

onnx2tf \
    -i /work/rtmw.onnx \
    -o /work/output \
    -osd \
    -b 1 \
    -ois input:1,3,384,288

echo ""
echo "=== Done ==="
ls -lh /work/output/*.tflite 2>/dev/null
ls -lh /work/output/saved_model/ 2>/dev/null
