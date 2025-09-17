# ✨ Handwritten Digit Recognition System 🖊️

**A web-based Handwritten Digit Recognition System built using TensorFlow and Flask.
Upload an image of a digit (0–9) and the system predicts the digit instantly using a Convolutional Neural Network (CNN) trained on the MNIST dataset.**


# 🌟 Features

**🧠 CNN model trained on MNIST dataset for high-accuracy digit recognition.**

**🌐 Flask web app for real-time digit prediction.**

**🎨 Image preprocessing:**

  1. Converts uploaded images to grayscale

  2. Crops and centers the digit

  3. Resizes to 28×28 pixels

  4. Normalizes pixel values for the model

**💻 Responsive frontend using Bootstrap**

**📊 Shows prediction probabilities for all digits (0–9)**


# 🚀 Installation & Setup
1. **Clone the repository**
   
        git clone https://github.com/AKSHAY1224/Handwritten-Digit-Recognition.git

        cd Handwritten-Digit-Recognition

2. **Create a virtual environment (recommended)**
 
       python -m venv venv
   
     **Windows**
   
        .\venv\Scripts\activate

     **macOS/Linux**

       source venv/bin/activate

3. **Install dependencies**

        pip install -r requirements.txt

4. **Train the model (optional, mnist_cnn.h5 included)**

        python train_model.py

5. **Run the Flask app**

        python app.py

6. **Open in browser 🌐**

     **go to:**

       http://127.0.0.1:5000
     **and upload a digit image.**


# 🎯 Usage

Click Choose File and upload a handwritten digit image.

Click Upload & Predict.

The system displays the Predicted digit 🖊️ and Probabilities for all digits (0–9) 📊


# 🛠️ Technologies Used

**Python 3 🐍**

**TensorFlow / Keras 🧠**

**Flask 🌐**

**Pillow (PIL) 🎨**

**Bootstrap 5 💻**

**HTML / CSS / JavaScript 🌟**


# 🌈 Future Improvements

**✍️ Draw-on-canvas feature for direct digit input in the browser**

**☁️ Deploy online (Heroku, Render, PythonAnywhere) for live demo**


# 👤 Author

**Akshay Pratap Singh**

**B.Tech CSE (AIML) | Noida Institute of Engineering and Technology**
