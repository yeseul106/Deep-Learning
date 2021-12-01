import tensorflow as tf

saved_model_dir = "C:/Users/82109/Documents/Deep Learning/convert_tflite/models/2018_12_17_22_58_35.h5"
# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# tflite_model = converter.convert()
#
# with open("converted_model.tflite", "wb") as f :
#     f.write(tflite_model)

from tensorflow import keras
model = keras.models.load_model(saved_model_dir, compile=False)

# .h5 -> .pb
export_path = 'C:/Users/82109/Documents/Deep Learning/convert_tflite/models/'
model.save(export_path, save_format="tf")

