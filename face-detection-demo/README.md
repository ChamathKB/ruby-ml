# face detection demo

face detection with onnx runtime converted `version-RFB-640.onnx` model using ruby `onnxruntime` and `mini_magick` gems.

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
ruby face_detection.rb 
```
this will generate labeled image in the directory and model info in terminal.