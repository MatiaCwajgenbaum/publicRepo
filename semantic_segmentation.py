# import pixellib
# from pixellib.semantic import semantic_segmentation
import cv2
import os
import tensorflow
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image as im

def do_semantic(img_path):
    # absolute_path = os.path.abspath("model/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    absolute_path = os.path.abspath("cce_dice_loss_pretrained_model.h5")
    # img = cv2.imread(img_path)
    img = mpimg.imread(img_path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)  # [:,:,0]

    # segment_image = semantic_segmentation()
    model = tf.keras.models.load_model(absolute_path, compile=False)
    x = img.reshape(1, 256, 256, 1)
    y = model.predict(x)
    # Load the xception model trained on pascal voc for segmenting objects.
    # segment_image.load_pascalvoc_model(absolute_path)
    # segment_image.load_pascalvoc_model("pascal.h5")

    # Perform segmentation on an image
    # segmap, segoverlay = segment_image.segmentAsPascalvoc(img_path, output_image_name="image1_new.jpg",
    #                                                       overlay=True)

    # return segoverlay
    return y[0, ...].argmax(axis=2), img

# output, segmap = segment_image.segmentAsPascalvoc("sample1.jpg")
# cv2.imwrite("img.jpg", segoverlay)
# print(segoverlay.shape)
