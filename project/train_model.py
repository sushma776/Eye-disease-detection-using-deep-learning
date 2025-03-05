import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set dataset directory paths
train_dir = r"C:\Users\chava\OneDrive\Desktop\EyeDiseaseDetection\Dataset\train"
test_dir = r"C:\Users\chava\OneDrive\Desktop\EyeDiseaseDetection\Dataset\test"

# Image Preprocessing
datagen = ImageDataGenerator(rescale=1./255)

# Load Training Data
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load Test Data
test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load Pre-trained Model (VGG16)
base_model = tf.keras.applications.VGG16(
    weights="imagenet", 
    include_top=False, 
    input_shape=(224, 224, 3)
)

# Freeze Pre-trained Layers
for layer in base_model.layers:
    layer.trainable = False

# Add Custom Layers
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation="softmax")  # 4 classes
])

# Compile Model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Print Model Summary
model.summary()

# Train Model
history = model.fit(
    train_data,
    epochs=10
)

# Evaluate on Test Data
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2f}")

# Save Trained Model
model.save("evgg.h5")
