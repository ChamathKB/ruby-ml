# Object detection demo

object detection with onnx runtime converted `ssd_mobilenet_v1_12` model using ruby `onnxruntime` and `mini_magick` gems.

### Convert model (optional)
if model is not in onnx format;

install `tf2onnx`
```
pip install tf2onnx
```

convert the model to onnx format
```
python -m tf2onnx.convert --opset 10 \
  --saved-model ssd_mobilenet_v1_coco_2018_01_28/saved_model \
  --output model.onnx
```

or run
```
tf2onnx_converter.sh
```
### Install ruby dependencies
add following to Gemfile and
```
gem "onnxruntime"
gem "mini_magick"
```
run
```
bundle install
```

### Make predictions
  
run
```
ruby object_detection.rb 
```
this will generate labeled image in the directory and model info in terminal.