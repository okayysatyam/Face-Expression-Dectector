import time
import numpy as np
import tensorflow as tf

# Load test image (replace with actual preprocessing/load code)
test_image = np.random.rand(1, 48, 48, 1).astype(np.float32)

# Load Keras model
keras_model = tf.keras.models.load_model('emotiondetector.keras')

# Keras inference timing
keras_times = []
for _ in range(100):
    start = time.time()
    keras_model.predict(test_image, verbose=0)  # Added verbose=0 to suppress output
    keras_times.append(time.time() - start)

print(f"Keras average inference time: {np.mean(keras_times)*1000:.2f} ms")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='emotion_detector_quantized.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# TFLite inference timing
tflite_times = []
for _ in range(100):
    interpreter.set_tensor(input_details[0]['index'], test_image)
    start = time.time()
    interpreter.invoke()
    tflite_times.append(time.time() - start)

print(f"TFLite average inference time: {np.mean(tflite_times)*1000:.2f} ms")

# Calculate reduction
reduction = (np.mean(keras_times) - np.mean(tflite_times)) / np.mean(keras_times) * 100
print(f"Inference time reduction: {reduction:.1f}%")
