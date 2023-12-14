# emotion detection demo

emotion detection with onnx runtime converted `FERPlus.onnx` model using ruby `onnxruntime` and `mini_magick` gems.

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
ruby emotion_detection.rb 
```
this will generate labeled image in the directory and model info in terminal.