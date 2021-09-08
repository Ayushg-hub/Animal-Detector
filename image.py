import tensorflow as tf
from Class_Names import class_names
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

    # extract the box dimensions
    x, y, width, height, confidence, classes = pred[:, :, 0:1], pred[:, :, 1:2], pred[:, :, 2:3], pred[:, :, 3:4], pred[:, :,4:5], pred[:,:,5:]
    scores = confidence * classes

    # convert box - coords into a suitable format for non-max-suppression
    top_l_x = x - width / 2
    bottom_r_x = x + width / 2
    top_l_y = y - width / 2
    bottom_r_y = y + width / 2

    # pack them again in one list
    boxes = tf.concat([top_l_x, top_l_y, bottom_r_x, bottom_r_y], axis=-1)

    # change their dimensions for NMS by tf
    batch_size = boxes.shape[0]
    num_boxes = boxes.shape[1]
    q = 1
    boxes = boxes / model_size[0]
    boxes = tf.reshape(boxes, (batch_size, num_boxes, q, 4))
    scores = tf.reshape(scores, (batch_size, num_boxes, num_classes))

    # now apply non max suppression
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(boxes=boxes,
                                                                                     scores=scores,
                                                                                     max_output_size_per_class=max_output_size_per_class,
                                                                                     max_total_size=max_output_size,
                                                                                     iou_threshold=iou_threshold,
                                                                                     score_threshold=confidence_threshold)

    image = np.squeeze(image)

    # draw boxes
    for i in range(valid_detections[0]):
        # imd dim are heightXwidth
        x1_y1 = tuple((int(boxes[0, i, 0:1] * image.shape[1]), int(boxes[0, i, 1:2] * (image.shape[0]))))
        x2_y2 = tuple((int(boxes[0, i, 2:3] * image.shape[1]), int(boxes[0, i, 3:4] * (image.shape[0]))))

        img = cv2.rectangle(image, x1_y1, x2_y2, (0, 255, 0), 2)

        img = cv2.putText(img,
                            "{} {:.4f}".format(class_names[int(classes[0, i])], scores[0, i]),
                            (x1_y1),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2)

    win_name = 'Image detection'

    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('test_result.jpg', img)

if __name__ == '__main__':
        main()