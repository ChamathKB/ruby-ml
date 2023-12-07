require "onnxruntime"
require "mini_magick"
require "numo/narray"
#require "chunky_png"
#require "opencv"


def load_model(model_path)
  OnnxRuntime::Model.new(model_path)
end

def get_image_pixels(image)
  image.get_pixels
end

def draw_box(img, box)
  width, height = img.dimensions

  thickness = 2
  top = (box[0] * height).round - thickness
  left = (box[1] * width).round - thickness
  bottom = (box[2] * height).round + thickness
  right = (box[3] * width).round + thickness

  img.combine_options do |c|
    c.draw "rectangle #{left},#{top} #{right},#{bottom}"
    c.fill "none"
    c.stroke "red"
    c.strokewidth thickness
  end

  img.combine_options do |c|
    c.fill "red"
    c.pointsize 18
  end
end

def predict(model, pixels)
  model.predict({"inputs" => [pixels]})
end

def label_image(img, result)
  result["num_detections"].each_with_index do |n, idx|
    n.to_i.times do |i|
      box = result["detection_boxes"][idx][i]
      draw_box(img, box)
    end
  end
end

def print_model_and_result_info(model, result)
  puts "Model Inputs: #{model.inputs}"
  puts "Model Outputs: #{model.outputs}"
  puts "Model Metadata: #{model.metadata}"
end

def save_labeled_image(img, output_path)
  img.write(output_path)
end

## With minimagic gem
def load_image(image_path)
  MiniMagick::Image.open(image_path)
end

def preprocess_image(image, target_shape)
  img = image.dup # Duplicate the image to avoid modifying the original
  img.resize("#{target_shape[3]}x#{target_shape[2]}!")
  img.format("RGB") # Ensure the image is in RGB format

  # Normalize pixel values to be in the range [0, 1]
  pixels = img.get_pixels.map { |pixel| pixel.map { |channel| channel / 255.0 } }

  # Transpose the data to match the expected shape [1, 3, 480, 640]
  pixels = Numo::NArray.cast(pixels).reshape(target_shape)

  # Add a batch dimension [1, 3, 480, 640]
  pixels = pixels.insert(0, 1)

  pixels
end

## With chunky_png gem
# def load_image(image_path)
#   ChunkyPNG::Image.from_file(image_path)
# end

# def preprocess_image(image, target_shape)
#   img = image.resize(target_shape[3], target_shape[2])
#   img.save("/tmp/preprocessed_image.png") # Save the preprocessed image for debugging

#   # Normalize pixel values to be in the range [0, 1]
#   pixels = img.pixels.map { |pixel| pixel / 255.0 }

#   # Transpose the data to match the expected shape [1, 3, 480, 640]
#   pixels = Numo::NArray.cast(pixels).reshape(target_shape)

#   # Add a batch dimension [1, 3, 480, 640]
#   pixels = pixels.insert(0, 1)

#   pixels
# end

## With OpenCV gem
# def load_image(image_path)
#     OpenCV::CvMat.load(image_path)
# end    

# def preprocess_image(orig_image, threshold = 0.7)
#   # Convert BGR to RGB
#   image = orig_image.bgr2rgb

#   # Resize the image to (640, 480)
#   image = image.resize(640, 480)

#   # Set image mean
#   image_mean = Numo::NArray[127, 127, 127]

#   # Normalize and transpose the image
#   image = (image - image_mean) / 128
#   image = image.transpose(2, 0, 1)

#   # Add batch dimension
#   image = image.expand_dims(0)

#   # Convert image to float32
#   image = image.astype(Numo::SFloat)

#   # Example usage:
#   # Assuming `orig_image` is an OpenCV image (CvMat) loaded using OpenCV, you can use the function like this:
#   # processed_image = face_detector(orig_image)
#   return image
# end

begin
  image_path = './data/faces.jpg'
  model_path = './models/version-RFB-640.onnx'
  output_path = 'detected.jpg'

  img = load_image(image_path)
  target_shape = [1, 3, 480, 640]
  pixels = preprocess_image(img, target_shape)

  model = load_model(model_path)
  result = predict(model, {"inputs" => pixels})

  print_model_and_result_info(model, result)
  label_image(img, result)
  save_labeled_image(img, output_path)
rescue StandardError => e
  puts "Error: #{e.message}"
end