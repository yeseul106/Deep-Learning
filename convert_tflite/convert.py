import tensorflow as tf
saved_model_dir = "C:/Users/82109/Documents/Deep Learning/convert_tflite/models/2018_12_17_22_58_35.h5"

# from tensorflow import keras
# model = keras.models.load_model(saved_model_dir, compile=False)
#
# # .h5 -> .pb
# export_path = 'C:/Users/82109/Documents/Deep Learning/convert_tflite/models/'
# model.save(export_path, save_format="tf")

# .pb -> .tflite
saved_model_dir = 'C:/Users/82109/Documents/Deep Learning/convert_tflite/models/'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('C:/Users/82109/Documents/Deep Learning/convert_tflite/models/converted_model.tflite', 'wb').write(tflite_model)
