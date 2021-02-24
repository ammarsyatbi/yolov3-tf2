import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import mask_outputs
import os

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', None, 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './masked_output.jpg', 'path to output image or directory')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('datadir', './data/test', 'folder containing images to detect')

# TODO: change flag to constants variable
def detection(model, img_raw, class_names):
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = model(img)
    t2 = time.time()
    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)

    img = mask_outputs(img, (boxes, scores, classes, nums), class_names)
    return img

def write_img(img_path, img):
    cv2.imwrite(img_path, img)
    logging.info('output saved to: {}'.format(img_path))

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
        detected_img = detection(yolo, img_raw, class_names )
        write_img(FLAGS.output, detected_img)

    elif FLAGS.image:
        img_raw = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
        detected_img = detection(yolo, img_raw, class_names )
        write_img(FLAGS.output, detected_img)

    else:
        os.makedirs(FLAGS.output, exist_ok=True)
        img_names = os.listdir(FLAGS.datadir)
        for img_name in img_names:
            img_path = os.path.join(FLAGS.datadir, img_name)
            img_raw = tf.image.decode_image(open(img_path, 'rb').read(), channels=3)
            detected_img = detection(yolo, img_raw, class_names )
            write_img(os.path.join(FLAGS.output,img_name ), detected_img)
    
        


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass