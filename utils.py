from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import pycocotools.mask as mask_util
from skimage.measure import find_contours

import boto3
from itertools import groupby
import random
import cv2
import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import tensorflow as tf
from colorthief import ColorThief
from sty import fg, bg, ef, rs
from PIL import Image
from matplotlib.gridspec import GridSpec
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
classes_segmentation = {'upperbody':0,
                        'lowerbody':1,
                        'wholebody':2,
                        'footwear':3,
                        'accessories':4}
classes_segmentation_names = ['upperbody',
                        'lowerbody',
                        'wholebody',
                        'footwear',
                        'accessories']
model_class = ['upperbody', 'lowerbody', 'wholebody', 'footwear', 'accessories']
main_directory = os.path.dirname(os.path.realpath(__file__))

def get_cfg(cfg_path,model_path):
    with open(cfg_path,'rb') as f:
        cfg = pickle.load(f)

    cfg.MODEL.WEIGHTS = os.path.join(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    #Tahmin edilecek sınıfların listesi
    MetadataCatalog.get("train").set(thing_classes=['upperbody', 'lowerbody', 'wholebody','footwear', 'accessories'])
    return cfg

def crop_object(image, box):
  """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
  """

  x_top_left = box[0]
  y_top_left = box[1]
  x_bottom_right = box[2]
  y_bottom_right = box[3]
  x_center = (x_top_left + x_bottom_right) / 2
  y_center = (y_top_left + y_bottom_right) / 2

  crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
  return crop_img

def get_classification_model(model_name):
    if model_name == "upperbody":
        #model_url = "https://fashionanalysis.s3.amazonaws.com/Modeller/upperbody_acc_train_092_val_073_incV3_299_299.h5"
        #local_path = tf.keras.utils.get_file("my-model.h5", model_url)
        return tf.keras.models.load_model(os.path.join(main_directory, 'modeller/upperbody/upperbody_acc_train_092_val_073_incV3_299_299.h5'))#(os.path.join(main_directory, 'modeller/upperbody/upperbody_acc_train_092_val_073_incV3_299_299.h5'))
    if model_name == "lowerbody":
        #model_url = "https://fashionanalysis.s3.amazonaws.com/Modeller/lowerbody_acc_train_095_val_087_incV3_299_299_v2.h5"
        #local_path = tf.keras.utils.get_file("my-model.h5", model_url)
        return tf.keras.models.load_model(os.path.join(main_directory, 'modeller/lowerbody/lowerbody_acc_train_095_val_087_incV3_299_299_v2.h5'))#(os.path.join(main_directory, 'modeller/lowerbody/lowerbody_acc_train_095_val_087_incV3_299_299_v2.h5'))
    if model_name == "pants":
        #model_url = "https://fashionanalysis.s3.amazonaws.com/Modeller/pants_type_acc_train_09_val_08_incV3_299_299.h5"
        #local_path = tf.keras.utils.get_file("my-model.h5", model_url)
        return tf.keras.models.load_model(os.path.join(main_directory, 'modeller/lowerbody/pants/pants_type_acc_train_09_val_08_incV3_299_299.h5'))#(os.path.join(main_directory, 'modeller/lowerbody/pants/pants_type_acc_train_09_val_08_incV3_299_299.h5'))
    if model_name == "accessories":
        #model_url = "https://fashionanalysis.s3.amazonaws.com/Modeller/accessories_acc_train_098_val_093_incV3_299_299_v2.h5"
        #local_path = tf.keras.utils.get_file("my-model.h5", model_url)
        return tf.keras.models.load_model(os.path.join(main_directory, 'modeller/accessories/accessories_acc_train_098_val_093_incV3_299_299_v2.h5'))#(os.path.join(main_directory, 'modeller/accessories/accessories_acc_train_098_val_093_incV3_299_299_v2.h5'))
    if model_name == "shirt_blouse":
        #model_url = "https://fashionanalysis.s3.amazonaws.com/Modeller/gomlek_kol_uzunluk_acc_train_099_val_089_incV3_299_299.h5"
        #local_path = tf.keras.utils.get_file("my-model.h5", model_url)
        return tf.keras.models.load_model(os.path.join(main_directory, 'modeller/upperbody/gomlek_kol/gomlek_kol_uzunluk_acc_train_099_val_089_incV3_299_299.h5'))#(os.path.join(main_directory, 'modeller/upperbody/gomlek_kol/gomlek_kol_uzunluk_acc_train_099_val_089_incV3_299_299.h5'))
    if model_name == "sweater":
        #model_url = "https://fashionanalysis.s3.amazonaws.com/Modeller/shirt_neck_acc_train_093_val_083_incV3_299_299_v3.h5"
        #local_path = tf.keras.utils.get_file("my-model.h5", model_url)
        return tf.keras.models.load_model(os.path.join(main_directory, 'modeller/upperbody/kazak_yaka/shirt_neck_acc_train_093_val_083_incV3_299_299_v3.h5'))#(os.path.join(main_directory, 'modeller/upperbody/kazak_yaka/shirt_neck_acc_train_093_val_083_incV3_299_299_v3.h5'))
    if model_name == "skirt_length":
        #model_url = "https://fashionanalysis.s3.amazonaws.com/Modeller/skirt_length_incV3_299_299.h5"
        #local_path = tf.keras.utils.get_file("my-model.h5", model_url)
        return tf.keras.models.load_model(os.path.join(main_directory, 'modeller/lowerbody/skirt_length/skirt_length_incV3_299_299.h5'))#(os.path.join(main_directory, 'modeller/lowerbody/skirt_length/skirt_length_incV3_299_299.h5'))   
    if model_name == "skirt_opening":
        #model_url = "https://fashionanalysis.s3.amazonaws.com/Modeller/skirt_opening_acc_train_099_val_093_incV3_299_299.h5"
        #local_path = tf.keras.utils.get_file("my-model.h5", model_url)
        return tf.keras.models.load_model(os.path.join(main_directory, 'modeller/lowerbody/skirt_opening/skirt_opening_acc_train_099_val_093_incV3_299_299.h5'))#(os.path.join(main_directory, 'modeller/lowerbody/skirt_opening/skirt_opening_acc_train_099_val_093_incV3_299_299.h5'))   

def get_model_classes(class_name):
    if class_name == "upperbody":
        return ['cardigan','jacket','shirt_blouse','sweater','top_t_shirt_sweatshirt','vest']
    if class_name == "lowerbody":
        return ['pants', 'shorts', 'skirt']
    if class_name == "pants":
        return ['capri', 'sailor', 'sweat']
    if class_name == "accessories":
        return ['bag, wallet', 'glasses', 'hat', 'tie', 'umbrella', 'watch']
    if class_name == "shirt_blouse":
        return ['short (length)', 'wrist-length']
    if class_name == "sweater":
        return ['round (neck)', 'turtle (neck)', 'v-neck']
    if class_name == "skirt_length":
        return ['maxi', 'midi', 'mini'] 
    if class_name == "skirt_opening":
        return ['fly (opening)', 'no opening']

def predict_class_name(image,class_name):
    prediction_scores = get_classification_model(class_name).predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    cls_name = get_model_classes(class_name)[predicted_index]
    return cls_name

def get_color(image):
    color_thief = ColorThief(image)
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=8)
    a = dominant_color[0]
    b = dominant_color[1]
    c = dominant_color[2]
    bar = bg(a,b,c) + str(a) + ","+ str(b) + "," + str(c) + bg.rs
    return [a,b,c]

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def print_classes(im_outputs,image_path):
    upperbody_list = []
    lowerbody_list= []
    accessories_list = []
    wholebody_list = []
    footwear_list = []
    
    upperbody_color = []
    lowerbody_color= []
    accessories_color = []
    wholebody_color = []
    footwear_color = []

    results = []

    for instances in range(len(im_outputs["instances"])):
        box = im_outputs["instances"][instances].pred_boxes.tensor.tolist()[0]
        mask = im_outputs["instances"][instances].pred_masks[0].cpu().numpy()
        cls_name = model_class[im_outputs["instances"][instances].pred_classes.tolist()[0]]
        image = Image.open(image_path)

        maks_2 = np.stack((mask,mask,mask),axis=2).astype(np.bool_)
        masked_img = image * maks_2
        cv2.imwrite('masked_img.png', cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)) 
        image = Image.open("masked_img.png")
        crop_img = crop_object(image, box)
        crop_img = np.array(crop_img)
        masked_num = 1
        crop_img[crop_img<masked_num] = 255
        crop_img = Image.fromarray(np.uint8(crop_img)).convert('RGB')
        image = np.asarray(crop_img.resize((299, 299)))/255.0

        filename = "Crop_"+str(instances)+".png"
        crop_img.save(filename)
        color_image = get_color(filename)
        #color_images.append(color_image)
        os.remove(filename)
        os.remove("masked_img.png")
        if cls_name == "upperbody":
            upperbody_classes = {"Category":"",
                        "Upperbody main category":"",
                        "Shirt-blouse category":"",
                        "Sweater category":""}
            color_image_class = {"upperbody "+str(instances):color_image}
            upperbody_classes["Category"] = cls_name
            upp_cls_name = predict_class_name(image,"upperbody")
            upperbody_classes["Upperbody main category"] = upp_cls_name
            if upp_cls_name == "shirt_blouse":
                upp_cls_name = predict_class_name(image,"shirt_blouse")
                upperbody_classes["Shirt-blouse sleeve length category"] = upp_cls_name
            if upp_cls_name == "sweater":
                upp_cls_name = predict_class_name(image,"sweater")
                upperbody_classes["Sweater category"] = upp_cls_name
            upperbody_list.append(upperbody_classes)
            upperbody_color.append(color_image_class)
        if cls_name == "lowerbody":
            lowerbody_classes = {"Category":"",
                        "Lowerbody main category":"",
                        "Pants category":"",
                        "Skirt-length category":"",
                        "Skirt-opening category":""}
            color_image_class = {"lowerbody "+str(instances):color_image}
            lowerbody_classes["Category"] = cls_name
            low_cls_name = predict_class_name(image,"lowerbody")
            lowerbody_classes["Lowerbody main category"] = low_cls_name
            if low_cls_name == "pants":
                low_cls_name = predict_class_name(image,"pants")
                lowerbody_classes["Pants category"] = low_cls_name
            if low_cls_name == "skirt":
                low_cls_name = predict_class_name(image,"skirt_length")
                low_cls_name_2 = predict_class_name(image,"skirt_opening")
                lowerbody_classes["Skirt-length category"] = low_cls_name
                lowerbody_classes["Skirt-opening category"] = low_cls_name_2
            lowerbody_list.append(lowerbody_classes)
            lowerbody_color.append(color_image_class)
        if cls_name == "accessories":
            accessories_classes = {"Category":"",
                        "Accessories main category":""}
            color_image_class = {"accessories "+str(instances):color_image}
            accessories_classes["Category"] = cls_name
            acc_cls_name = predict_class_name(image,"accessories")
            accessories_classes["Accessories main category"] = acc_cls_name
            accessories_list.append(accessories_classes)
            accessories_color.append(color_image_class)
        if cls_name == "wholebody":
            wholebody_classes = {"Category":""}
            color_image_class = {"wholebody "+str(instances):color_image}
            wholebody_classes["Category"] = cls_name
            wholebody_list.append(wholebody_classes)
            wholebody_color.append(color_image_class)
        if cls_name == "footwear":
            footwear_classes = {"Category":""}
            color_image_class = {"footwear "+str(instances):color_image}
            footwear_classes["Category"] = cls_name
            footwear_list.append(footwear_classes)
            footwear_color.append(color_image_class)

    classes = [upperbody_list,lowerbody_list,accessories_list,wholebody_list,footwear_list]
    color_classes = [upperbody_color,lowerbody_color,accessories_color,wholebody_color,footwear_color]
    classes_names = ["upperbody","lowerbody","accessories","wholebody","footwear"]
    
    ind_class_name = 0
    for ind,fashion_classes in enumerate(classes):
        sayac = 0
        for fashion_class in fashion_classes:
            if {k: v for k, v in fashion_class.items() if v} != {}:
                a,b,c = list(color_classes[ind][sayac].items())[0][1]
                print({k: v for k, v in fashion_class.items() if v}," -> Dominant color : ",bg(a,b,c) + str(a) + ","+ str(b) + "," + str(c) + bg.rs)
                results.append([{k: v for k, v in fashion_class.items() if v}," -> Dominant color : ",list(color_classes[ind][sayac].items())[0][1]])
                sayac = sayac + 1
    return results

def predict_image(image_path, predictor, cfg,save_dir=None, show=False):
    r"""
    Verilen resimdeki kıyafetlerin analizini yapıp sonuçlarını gösterir.

    Inputs:

        image_path = Resmin yolu

        predictor = Detectron2 modelinin yolu

        cfg = Detectron2 modelinin konfigürasyon değişkeni

        save_dir = Sonuçların(Detectron2 model sonucu,Genel analiz sonucu,Sınıflandırma modellerinin sonucu) 
                   kaydedilmesi.

        show = Genel analiz sonucunun gösterilmesi

    """
    instance_number = 0
    ext = 0.0
    while tqdm(instance_number == 0):
        print(instance_number)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80 - ext
        #MetadataCatalog.get("train").set(thing_classes=['upperbody', 'lowerbody', 'footwear', 'accessories'])
        predictor = DefaultPredictor(cfg)
        im = cv2.imread(image_path)
        outputs = predictor(im)
        instance_number = len(outputs["instances"])
        ext = ext + 0.05
    v = Visualizer(im[:,:,::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.9, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    results = print_classes(outputs,image_path)
    fig = plt.figure(figsize=(10, 15))#fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(14,10))
    gs = GridSpec(nrows=2, ncols=2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax0.axis('off')
    
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(v.get_image())
    ax1.axis('off')
    text = ""
    colors = []
    for i in results:
        a,b,c = i[2][0],i[2][1],i[2][2]
        text += str(i[0])+str(i[1])+str(a) + ","+ str(b) + "," + str(c)+ '\n'
        colors.append([a,b,c])
    colors = np.array(colors)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.imshow(colors.reshape(1,len(colors),3)/255.)
    fig.text(.5, .05, text, ha='center')
    fig.tight_layout()
    if show:
        plt.show()
    if save_dir != None:
        s3 = boto3.client('s3')
        transfer = boto3.s3.transfer.S3Transfer(s3)
        newpath = os.path.join(save_dir,image_path.split("/")[-1].split(".")[0])
        print(newpath)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        fig.savefig(os.path.join(newpath, image_path.split("/")[-1].split(".")[0]+'_result.png'))
        filename = os.path.join(newpath, image_path.split("/")[-1].split(".")[0]+'.png') 
        with open(os.path.join(newpath, image_path.split("/")[-1].split(".")[0]+".txt"), "w") as output:
            output.write(str(results))
        result_image_name = os.path.join(newpath, image_path.split("/")[-1].split(".")[0]+'_result.png')
        cv2.imwrite(filename, cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR))
        transfer.upload_file(filename, 'fashionanalysis', 'Orijinal_resimler/'+image_path.split("/")[-1].split(".")[0]+'.png')
        transfer.upload_file(result_image_name, 'fashionanalysis', 'Tahmin_sonuclari/'+image_path.split("/")[-1].split(".")[0]+'_result.png')
        result_url = "https://fashionanalysis.s3.amazonaws.com/Tahmin_sonuclari/"+ image_path.split("/")[-1].split(".")[0]+'_result.png'
    return results,result_url

def predict_image_detectron(image_path, predictor, cfg,save_dir=None, show=False):
    r"""
    Verilen resimdeki kıyafetlerin analizini yapıp sonuçlarını gösterir.

    Inputs:

        image_path = Resmin yolu

        predictor = Detectron2 modelinin yolu

        cfg = Detectron2 modelinin konfigürasyon değişkeni

        save_dir = Sonuçların(Detectron2 model sonucu,Genel analiz sonucu,Sınıflandırma modellerinin sonucu) 
                   kaydedilmesi.

        show = Genel analiz sonucunun gösterilmesi

    """
    instance_number = 0
    ext = 0.0
    while tqdm(instance_number == 0):
        print(instance_number)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80 - ext
        #MetadataCatalog.get("train").set(thing_classes=['upperbody', 'lowerbody', 'footwear', 'accessories'])
        predictor = DefaultPredictor(cfg)
        im = cv2.imread(image_path)
        outputs = predictor(im)
        instance_number = len(outputs["instances"])
        ext = ext + 0.05
    v = Visualizer(im[:,:,::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.9, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    results = str([classes_segmentation_names[i] for i in outputs["instances"].pred_classes.cpu().numpy()])
    print(results)
    s3 = boto3.client('s3')
    transfer = boto3.s3.transfer.S3Transfer(s3)

    newpath = os.path.join(save_dir,image_path.split("/")[-1].split(".")[0])
    print(newpath)
    if not os.path.exists(newpath):
        os.mkdir(newpath)
    filename = os.path.join(newpath, image_path.split("/")[-1].split(".")[0]+'_result.png') 
    print("filename:",filename)
    #shutil.copy(src, dst)
    with open(os.path.join(newpath, image_path.split("/")[-1].split(".")[0]+".txt"), "w") as output:
        output.write(str(results))
    result_image_name = os.path.join(newpath, image_path.split("/")[-1].split(".")[0]+'.png')
    cv2.imwrite(filename, cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR))
    transfer.upload_file(image_path, 'fashionanalysis', 'Orijinal_resimler/'+image_path.split("/")[-1].split(".")[0]+'.png')
    transfer.upload_file(filename, 'fashionanalysis', 'Tahmin_sonuclari/'+image_path.split("/")[-1].split(".")[0]+'_result.png')
    result_url = "https://fashionanalysis.s3.amazonaws.com/Tahmin_sonuclari/"+ image_path.split("/")[-1].split(".")[0]+'_result.png'
    
    print(result_url)
    return results,result_url