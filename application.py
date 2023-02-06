from flask import Flask, render_template, request
import requests
import json
import os
import sys, os
main_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(main_dir+"/detectron2")
sys.path.append(main_dir+"/detectron2/detectron2")
from detectron2.config import get_cfg
import pickle
print(main_dir+"/detectron2")
print(main_dir+"/detectron2/detectron2")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from detectron2.engine import DefaultPredictor
#from detectron2.data import MetadataCatalog
from utils import *
CUDA_VISIBLE_DEVICES=1

main_dir = os.path.dirname(os.path.realpath(__file__))
#data_dir = os.path.join(main_dir,"data") # Tahmin edilecek resimin bulunduğu klasor
#save_dir = os.path.join(main_dir,"results") # Tahmin sonuçlarının kaydedileceği klasor
models_dir = os.path.join(main_dir,"modeller") # Model dosyalarının bulunduğu klasor

cfg_path = os.path.join(models_dir, "SE_CFG_3.pickle") # Modelin pickle dosyasının ismi
model_path = os.path.join("https://fashionanalysis.s3.amazonaws.com/Modeller/model_final.pth") # Model dosyasının ismi

#cfg_save_path = "/home/zoidata/Documents/Detectron2_Fashion/OD_lisance.pickle"
print(model_path)
with open(cfg_path,'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(model_path)#"Modeller"+.split(".")[-1]
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

#Tahmin edilecek sınıfların listesi
MetadataCatalog.get("train").set(thing_classes=['upperbody', 'lowerbody', 'wholebody','footwear', 'accessories'])
predictor = DefaultPredictor(cfg)

#cfg = get_cfg(cfg_path,model_path)
#predictor = DefaultPredictor(cfg)


application = Flask(__name__)

# routes
@application.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@application.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        bucket = "fashion-images-v1"
        img_path = "static/" + img.filename
        filename = img.filename
        img.save(img_path)
        print(img_path)
        result,pred_path = predict_image_detectron(img_path, predictor, cfg,save_dir=os.path.join(main_dir,"results"),show=False)
        print(pred_path)
    return render_template("index.html", prediction=result, img_path=img_path, pred_path = pred_path)

if __name__ == '__main__':
    # app.debug = True
    application.run(debug=True)
