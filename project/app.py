import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

model = load_model("evgg.h5")  # Load trained deep learning model
app = Flask(__name__)  # Initialize Flask app

@app.route('/')
def index():
    return render_template('index.html')  # Load home page

@app.route('/inp')
def inp():
    return render_template("img_input.html")  # Load image upload page

@app.route('/about')
def about():
    return render_template("about.html")  # Load About Us page ✅

@app.route('/predict', methods=['POST'])
def res():
    if request.method == 'POST':
        f = request.files['image']  # Get uploaded image
        filepath = os.path.join('uploads', f.filename)  # Save image
        f.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224,224,3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # Make prediction
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        result = str(index[prediction[0]])

        return render_template('output.html', prediction=result)

# This should be at the outermost level, not inside a function!
if __name__ == "__main__":
    app.run(debug=True)
