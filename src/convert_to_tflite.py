import tensorflow as tf

# Load your Keras model from .h5 file
model = tf.keras.models.load_model('emotiondetector.keras')

# Set up the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Optional: converter.target_spec.supported_types = [tf.float16]

# Convert and save the model
tflite_model = converter.convert()
with open('emotion_detector_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model successfully saved as emotion_detector_quantized.tflite")
