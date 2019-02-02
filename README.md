# Image-Segmentation

| Geenrated Video	|  Picture    |  Predicted    |
|:-----------:|:----------:|:---------:|
| [![Video](https://img.youtube.com/vi/cxAuoHRf1z4/0.jpg)](https://www.youtube.com/watch?v=cxAuoHRf1z4)   	| [![Introduction video](https://img.youtube.com/vi/mj32wCefQnE/0.jpg)](https://www.youtube.com/watch?v=mj32wCefQnE)| [![Introduction video](https://img.youtube.com/vi/piNMVkYRZwA/0.jpg)](https://www.youtube.com/watch?v=piNMVkYRZwA) | 

Keras impementation of BiseNet Image Segmentation Model (Paper : [Link](https://arxiv.org/pdf/1808.00897.pdf))

Pretrained Model Download ([Link](https://drive.google.com/uc?id=11ghYNpY4osChcteBV-fefqY8ufDjhcrq&export=download))

**Prediction supported file formats (Video : Mp4, Picture : .png)**


# Model prediction arguments

```
mandatory arguments:
  -media MEDIA_DIR, --media_dir MEDIA_DIR
                        Media Directorium for prediction (mp4,png)
optional arguments:
  -save SAVE_DIR, --save_dir SAVE_DIR
                        Save Directorium
  -model MODEL_DIR, --model_dir MODEL_DIR
                        Model Directorium
```

## Example Image Prediction : 

```
python predict.py -media test_img.png
```

## Example Video Prediction :
```
python predict.py -media test_video.mp4
```


# Model training arguments:

```
optional arguments:
  -eph EPOCHS, --epochs EPOCHS
                        Number of epochs
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate
  -save MODEL_DIR, --model_dir MODEL_DIR
                        Save checkpoints directory
  -batches BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of batches per train
```                        




