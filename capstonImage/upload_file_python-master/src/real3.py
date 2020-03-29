import os
import cv2
import numpy as np

from Sketcher import Sketcher
from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet

import sys
from copy import deepcopy


from flask import Flask, request, render_template, send_from_directory,redirect, url_for

__author__ = 'ibininja'

app = Flask(__name__)



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST","GET"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)

        upload.save(destination)
        #destination2 = destination.split("\\")
        #print(destination2)
        #destination2 = "/".join(destination2)
        #destination2 = destination2.replace("//","/")
        #blurring_image = filename + "_blurring.png"
    # return send_from_directory("images", filename, as_attachment=True)
    return redirect(url_for('blurring', image_name=filename, destination=destination,filename = filename )) #destination2=destination2) )
    #return redirect(url_for('send_image', image_name=blurring_image))


#
# @app.route('/blurring')
# def blurring():
#



@app.route('/blurring/<filename>')
def blurring(filename):
    ##
    filename2 ="C:/Users/admin/PycharmProjects/capstonImage/upload_file_python-master/src/images/" +filename
    ##
    print('load model...')
    model = PConvUnet(vgg_weights=None, inference_only=True)
    model.load('pconv_imagenet.h5', train_bn=False)
    # model.summary()
    #print("send_image_destination", {destination2} )
    # img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    #destination="C%3A%5CUsers%5Cadmin%5CPycharmProjects%5CcapstonImage%5Cupload_file_python-master%5Csrc%5Cimages%2F%2Fresult.JPG"
    #destination="C:/Users/admin/PycharmProjects/capstonImage/upload_file_python-master/src/images/result.JPG"
    img = cv2.imread(filename2, cv2.IMREAD_COLOR)

    img_masked = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)

    sketcher = Sketcher('image', [img_masked, mask], lambda: ((255, 255, 255), 255))
    chunker = ImageChunker(512, 512, 30)

    while True:
        key = cv2.waitKey()

        if key == ord('q'):  # quit
            break
            cv2.destroyAllWindows()
        if key == ord('r'):  # reset
            print('reset')
            img_masked[:] = img
            mask[:] = 0
            sketcher.show()
        if key == 32:  # hit spacebar to run inpainting
            input_img = img_masked.copy()
            input_img = input_img.astype(np.float32) / 255.

            input_mask = cv2.bitwise_not(mask)
            input_mask = input_mask.astype(np.float32) / 255.
            input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1)

            # cv2.imshow('input_img', input_img)
            # cv2.imshow('input_mask', input_mask)

            print('processing...')

            chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
            chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask))

            # for i, im in enumerate(chunked_imgs):
            #   cv2.imshow('im %s' % i, im)
            #   cv2.imshow('mk %s' % i, chunked_masks[i])

            pred_imgs = model.predict([chunked_imgs, chunked_masks])
            result_img = chunker.dimension_postprocess(pred_imgs, input_img)

            print('completed!')

            cv2.imshow('result', result_img)
            result_img22 = result_img.astype(np.float32) * 255
            #cv2.imwrite(destination[:-4]+'blur'+'.png', result_img22)
            cv2.imwrite(filename2[:-4]+"_blurring.png",result_img22)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

    filename = filename[:-4] + "_blurring.png"
    #return send_from_directory("images", filename+"_blurring.png")
    #return send_from_directory("images", blurring_image)
    return redirect(url_for('send_image',filename=filename))
@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)