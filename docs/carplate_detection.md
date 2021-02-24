# Carplate Detection

The carplate detection was trained on various open dataset obtain mostly from Kaggle. There are few preprocessing done into it but are not included in this repo. I will include once I've made proper modification to make it work with most dataset.

The link to trained carplate detection is [here](https://drive.google.com/file/d/1zy73fVk3k8_o28tvsHXlMid66eicnYel/view?usp=sharing)

# Results

This model was trained for 100 epoch, I disabled early stopping to see how much I can push the regression. Obviously not much improvement has been, this is due the lack of dataset I'm using.

# Train / Val Loss

# Test Images
Before: 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/IMG_20210124_204438.jpg)
After: 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/IMG_20210124_204438.jpg)

Before (Close up angled): 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/IMG_20210224_062813.jpg)
After (Close up angled): 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/IMG_20210224_062813.jpg)

Before (Noised with grill): 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/IMG_20210224_062808.jpg)
After (Noised with grill): 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/IMG_20210224_062808.jpg)

Before (Single car - front): 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/singlecar.jpg)
After (Single car - front): 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/singlecar.jpg)

Before (Multiple car): 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/multicar.jpg)
After (Multiple car): 
![alt text](https://github.com/ammarsyatbi/yolov3-tf2/blob/master/data/results/before/multicar.jpg)

# How to use

1. Download model [here](https://drive.google.com/file/d/1zy73fVk3k8_o28tvsHXlMid66eicnYel/view?usp=sharing)
2. Extract model in ./data/model/carplate
3. Run following command with your parameter (data directory and output directory)

```
python3.7 mask_detection.py \
	--classes ./data/carplate.names \
	--num_classes 1 \
	--weights ./data/model/carplate/yolov3_train_100.tf \
	--datadir ./data/images/test \
	--output ./data/images/masked
```

# Keytakes

 If given enough dataset, it can be improve to work better on different angle and multiple detection. The model seems work fine with different environment. For example, different lighting for days and night. All in all, yolov3 works well for car plate detection, a better alternative for this model would be yolov4. However, the reason I'm not using it is because I'm unable to fix certain tf2 issues.

# Future improvements

- Reorganize data directory
- Include image augmentation in training
- Detect from directory (now only available for masked detection)
- Improvise generate tf records for custom dataset

