# Command with my parameter 
  

# Tools to generate tfrecord from VOC
```
python3.7 tools/generate_tfrecords.py \
--data_dir '/home/mars/dataset/car_plate/' \
--split train \
--output_file ./data/carplate_train.tfrecord
```
```
python3.7 tools/generate_tfrecords.py \
--data_dir '/home/mars/dataset/car_plate/' \
--split test \
--output_file ./data/carplate_test.tfrecord
```

# Training 
```
python3.7 train.py \
--dataset ./data/carplate_train.tfrecord \
--val_dataset ./data/carplate_test.tfrecord \
--classes ./data/carplate.names \
--num_classes 1 \
--mode fit --transfer darknet \
--batch_size 4 \
--epochs 100 \
--weights ./checkpoints/yolov3.tf \
--weights_num_classes 80 \
--run carplate
```

## detect from images
```
python3.7 detect.py \
	--classes ./data/carplate.names \
	--num_classes 1 \
	--weights ./checkpoints/run/carplate/yolov3_train_100.tf \
	--image /home/mars/dataset/images/IMG_20210124_204438.jpg
```
## detect from validation set
```
python3.7 detect.py \
	--classes ./data/carplate.names \
	--num_classes 1 \
	--weights ./checkpoints/run/carplate/yolov3_train_100.tf \
	--tfrecord ./data/carplate_test.tfrecord
```
## masked detected image
```
python3.7 mask_detection.py \
	--classes ./data/carplate.names \
	--num_classes 1 \
	--weights ./checkpoints/run/carplate/yolov3_train_100.tf \
	--image /home/mars/dataset/images/IMG_20210124_204438.jpg
```

## masked detected image from folder
```
python3.7 mask_detection.py \
	--classes ./data/carplate.names \
	--num_classes 1 \
	--weights ./checkpoints/run/carplate/yolov3_train_100.tf \
	--datadir ./data/images/test \
	--output ./data/images/masked
```


	