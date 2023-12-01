#!/bin/bash
python -m tf2onnx.convert --opset 10 \
  --saved-model ssd_mobilenet_v1_coco_2018_01_28/saved_model \
  --output model.onnx