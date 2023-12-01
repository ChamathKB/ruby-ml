require "onnxruntime"
require "mini_magick"

def load_image(image_path)
  MiniMagick::Image.open(image_path)
end

def load_model(model_path)
  OnnxRuntime::Model.new(model_path)
end

def get_image_pixels(image)
  image.get_pixels
end

def draw_box(img, label, box)
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
    c.draw "text #{left},#{top - 5} \"#{label}\""
    c.fill "red"
    c.pointsize 18
  end
end

def predict(model, pixels)
  model.predict({"inputs" => [pixels]})
end

def label_image(img, result, coco_labels)
  result["num_detections"].each_with_index do |n, idx|
    n.to_i.times do |i|
      label = result["detection_classes"][idx][i].to_i
      label = coco_labels[label] || label
      box = result["detection_boxes"][idx][i]
      draw_box(img, label, box)
    end
  end
end

def print_model_and_result_info(model, result)
  puts "Model Inputs: #{model.inputs}"
  puts "Model Outputs: #{model.outputs}"
  puts "Model Metadata: #{model.metadata}"

  puts "Number of Detections: #{result["num_detections"]}"
  puts "Detection Classes: #{result["detection_classes"]}"
end

def save_labeled_image(img, output_path)
  img.write(output_path)
end

# Main program

image_path = './data/sample.jpg'
model_path = './models/ssd_mobilenet_v1_12.onnx'
output_path = "labeled.jpg"

img = load_image(image_path)
pixels = get_image_pixels(img)

model = load_model(model_path)
result = predict(model, pixels)

coco_labels = {
  23 => "bear",
  88 => "teddy bear"
}

print_model_and_result_info(model, result)
label_image(img, result, coco_labels)
save_labeled_image(img, output_path)
