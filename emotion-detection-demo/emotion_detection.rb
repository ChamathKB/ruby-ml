require "onnxruntime"
require "mini_magick"

def load_image(image_path)
    img = MiniMagick::Image.open("ranger.jpg")
    img.crop "100x100+60+20", "-gravity", "center"
    img.resize "64x64^", "-gravity", "center", "-extent", "64x64"
    img.colorspace "Gray"
    img.write("resized.jpg")
end

def get_image_pixels(img)
    # all pixels are the same for grayscale, so just get one of them
    pixels = img.get_pixels.flat_map { |r| r.map(&:first) }
    pixels = OnnxRuntime::Utils.reshape(pixels, [1, 1, 64, 64])
end

def load_model(model_path)
    model = OnnxRuntime::Model.new(model_path)
end

def predict(model, pixels)
    result = model.predict("Input3" => pixels)
end

def softmax(x)
  exp = x.map { |v| Math.exp(v - x.max) }
  exp.map { |v| v / exp.sum }
end


# Main program


image_path = './data/sample.jpg'
model_path = './models/model.onnx'

img = load_image(image_path)
pixels = get_image_pixels(img)

model = load_model(model_path)
result = predict(model, pixels)

probabilities = softmax(result["Plus692_Output_0"].first)

emotion_labels = [
  "neutral", "happiness", "surprise", "sadness",
  "anger", "disgust", "fear", "contempt"
]

pp emotion_labels.zip(probabilities).sort_by { |_, v| -v }.to_h