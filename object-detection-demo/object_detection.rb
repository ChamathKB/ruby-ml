require "onnxruntime"
require "mini_magick"

image_path = './data/sample.jpg'
model_path = './models/ssd_mobilenet_v1_12.onnx' 

img = MiniMagick::Image.open(image_path)
pixels = img.get_pixels

model = OnnxRuntime::Model.new(model_path)
result = model.predict({"inputs" => [pixels]})

p model.inputs
p model.outputs
p model.metadata

p result["num_detections"]
p result["detection_classes"]

coco_labels = {
  23 => "bear",
  88 => "teddy bear"
}

def draw_box(img, label, box)
  width, height = img.dimensions

  thickness = 2
  top = (box[0] * height).round - thickness
  left = (box[1] * width).round - thickness
  bottom = (box[2] * height).round + thickness
  right = (box[3] * width).round + thickness

  # draw box
  img.combine_options do |c|
    c.draw "rectangle #{left},#{top} #{right},#{bottom}"
    c.fill "none"
    c.stroke "red"
    c.strokewidth thickness
  end

  # draw text
  img.combine_options do |c|
    c.draw "text #{left},#{top - 5} \"#{label}\""
    c.fill "red"
    c.pointsize 18
  end
end

result["num_detections"].each_with_index do |n, idx|
  n.to_i.times do |i|
    label = result["detection_classes"][idx][i].to_i
    label = coco_labels[label] || label
    box = result["detection_boxes"][idx][i]
    draw_box(img, label, box)
  end
end

img.write("labeled.jpg")
