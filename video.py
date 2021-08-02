import tensorflow as tf
from utils import class_names,output_boxes,draw_outputs
from YOLO import YOLOv3Net
import cv2
import time

model_size = (416, 416,3)
num_classes = 80
max_output_size = 100
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5
weightfile = 'weights/yolov3_weights.tf'
video_path = './data/video/traffic_2.mp4'

def main():
    model = YOLOv3Net(model_size)
    model.load_weights(weightfile)

    window = 'object detection'
    cv2.namedWindow(window)

    cap = cv2.VideoCapture(video_path)

    # frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            start = time.time()
            ret,frame = cap.read()

            if not ret:
                break

            resized_frame = tf.expand_dims(frame,0)
            resized_frame = tf.image.resize(resized_frame,(model_size[0],model_size[1]))

            pred = model.predict(resized_frame)

            boxes,scores,classes,nums = output_boxes(pred,model_size,max_output_size,max_output_size_per_class,iou_threshold,confidence_threshold)

            img = draw_outputs(frame,boxes,scores,classes,nums,class_names)

            cv2.imshow(window,img)

            stop = time.time()

            time_elapsed = stop - start

            fps = 1/time_elapsed

            print("FPS : ",fps)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                  break
    finally:
        cv2.destroyAllWindows()
        cap.release()
        print("detection performed successfully !")

if __name__ == '__main__':
    main()




