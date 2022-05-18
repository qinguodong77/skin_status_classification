# 创建应用实例
import sys

from wxcloudrun import app

import json
from flask import Flask,make_response,render_template,request,url_for
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import models
from pre_ import pre_process,hisEqulColor2
import cv2 as cv
from models import resnet
import numpy as np
import os
import torch.nn as nn


# # 获取图片
# def get_image(ImageFilePath):
#     input_image=cv.imread(ImageFilePath)
#     return input_image


# 图像预处理
def pretreatment(strFilePath):
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor()])
    predimg = pre_process(transform)
    img_chw=predimg(strFilePath)
    return img_chw


def predict_image(model,ImageFilePath):

    input_image=ImageFilePath
    img_chw=pretreatment(input_image)
    if torch.cuda.is_available():
        img_chw = img_chw.to("cuda")
        model.to("cuda")
    input_list = [img_chw]
    # 不计算梯度输出
    with torch.no_grad():
        output_list = model(input_list)
        out_dict = output_list[0]
        return out_dict

app = Flask(__name__)

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x


with open('./static/model/classes.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    print("oldlabels:",labels)
    labels = list(map(lambda x: x.strip().split('\t'), labels))
    print("newlabels:",labels)


def padding_black(img):
    w, h = img.size

    scale = 224. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

    size_fg = img_fg.size
    size_bg = 224

    img_bg = Image.new("RGB", (size_bg, size_bg))

    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

    img = img_bg
    return img

@app.route('/',methods=['GET'])
def hello_world():
    MyName="abc"
    res=make_response(render_template('index.html',mName=MyName))
    return res

@app.route('/uploadImg',methods=['GET','POST'])
def uploadImg():
    pred_id=1
    if request.method=='GET':
        res= make_response(render_template("index.html"))
        return res
    elif request.method=='POST':
        if 'file' in request.files:
            objFile=request.files.get('file')
            strFileName=objFile.filename
            strFilePath="./static/myImages"+"/"+strFileName
            objFile.save(strFilePath)
            # 预测并返回
            objImg=Image.open(strFilePath).convert("RGB")
            preprocess = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ToTensor(),
            ])
            img_chw = preprocess(objImg)
            image=img_chw.unsqueeze(0)

            model = models.resnet(pretrained=False)
            fc_inputs = model.fc.in_features
            model.fc = nn.Linear(fc_inputs, 214)
            # model = model.cuda()
            # 加载训练好的模型
            # 导入模型
            print("model loading")

            path_model = "./static/model/resnet_ckpt.pt"
            model = resnet()
            model.load_state_dict(torch.load(path_model), False)
            # model = model_loaded.load_state_dict(torch.load(path_model))

            print("finish")

            src = image.numpy()
            src = src.reshape(3, 224, 224)
            src = np.transpose(src, (1, 2, 0))
            # image = image.cuda()
            # label = label.cuda()

            pred = model(image)
            pred = pred.data.cpu().numpy()[0]

            score = softmax(pred)
            pred_id = np.argmax(score)

            print('预测结果：', labels[pred_id][0])#将预测结果传回给前端
        return json.dumps(labels[pred_id][0], ensure_ascii=False)




if __name__ == '__main__':

    # host:任何机器都可以访问，port:端口号
    app.run(host='0.0.0.0',port=80,debug=True)


# http 80 tcp  https 443(安全) tcp/ip
# get post request response
# get request->服务器处理->浏览器渲染html内容




# 启动Flask Web服务
# if __name__ == '__main__':
#     app.run(host=sys.argv[1], port=sys.argv[2])
