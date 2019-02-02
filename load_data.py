import os
import cv2

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image

def get_data():

    PATH = os.getcwd()
    train_path = PATH+'\data\\train\\'
    trainy_path = PATH+'\data\\train_labels\\'

    val_path = PATH+'\data\\val\\'
    valy_path = PATH+'\data\\val_labels\\'


    train_batch = os.listdir(train_path)
    trainy_batch = os.listdir(trainy_path)

    val_batch = os.listdir(val_path)
    valy_batch = os.listdir(valy_path)

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    # if data are in form of images
    for sample in train_batch:
        img_path = train_path+sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(608, 608))
        X_train.append(x)

    for sample in trainy_batch:
        img_path = trainy_path+sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(608, 608))
        y_train.append(x)

    for sample in val_batch:
        img_path = val_path+sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(608, 608))
        X_val.append(x)

    for sample in valy_batch:
        img_path = valy_path+sample
        x = load_image(img_path)
        x = cv2.resize(x, dsize=(608, 608))
        y_val.append(x)

    return X_train, y_train, X_val, y_val
