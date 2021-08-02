import tensorflow as tf
from utils import class_names, output_boxes, draw_outputs
import cv2
import numpy as np
from YOLO import YOLOv3Net

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_size = (416, 416,3)
num_classes = 80
class_name = './data/coco.names'
max_output_size = 60
max_output_size_per_class= 40
iou_threshold = 0.5
confidence_threshold = 0.5
weightfile = 'weights/yolov3_weights.tf'
img_path = "data/images/test.jpg"

def main():

    model = YOLOv3Net(model_size)
    model.load_weights(weightfile)

    image = cv2.imread(img_path)
    image = np.array(image)
    image = tf.expand_dims(image, 0)
    resized_frame = tf.image.resize(image, (model_size[0],model_size[1]))

    pred = model.predict(resized_frame)
    boxes, scores, classes, nums = output_boxes( \
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold)

    image = np.squeeze(image)
    img = draw_outputs(image, boxes, scores, classes, nums, class_names)
    win_name = 'Image detection'

    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('test_result.jpg', img)

if __name__ == '__main__':
        main()