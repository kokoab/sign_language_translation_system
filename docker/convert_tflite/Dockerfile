FROM python:3.10-slim

RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnx2tf==1.26.3 \
    tensorflow==2.15.1 \
    tf_keras==2.15.1 \
    onnx_graphsurgeon==0.5.2 \
    sng4onnx==1.0.4 \
    flatbuffers==24.3.25 \
    psutil

WORKDIR /work
CMD ["bash"]
