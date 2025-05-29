from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key in production

# Load your trained model
model = load_model('Xception_model.h5')

# Diabetic retinopathy class labels
class_labels = ['mild', 'moderate', 'NO DR', 'Proliferate_DR', 'severe']

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory storage
users = {}      # key: username, value: password
patients = {}   # key: username, value: list of patient dicts

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['name'].strip()
        password = request.form['password'].strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not username or not password or not confirm_password:
            flash('Please fill out all fields.', 'danger')
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('signup'))

        if username in users:
            flash('Username already exists. Please choose another.', 'danger')
            return redirect(url_for('signup'))

        users[username] = password
        patients[username] = []
        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if not username or not password:
            flash('Please enter username and password.', 'danger')
            return redirect(url_for('login'))

        if username not in users:
            flash('User does not exist.', 'danger')
            return redirect(url_for('login'))

        if users[username] != password:
            flash('Wrong password.', 'danger')
            return redirect(url_for('login'))

        session['username'] = username
        flash('Logged in successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please login to access dashboard.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        flash('Please login first.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        patient_name = request.form['patient_name'].strip()
        file = request.files.get('image')

        if not patient_name:
            flash('Please enter patient name.', 'danger')
            return redirect(request.url)

        if not file or file.filename == '':
            flash('No selected file.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Preprocess image
        img = image.load_img(save_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        preds = model.predict(img_array)
        predicted_class = class_labels[np.argmax(preds)]

        # Save patient info
        patient_info = {
            'name': patient_name,
            'image': filename,
            'prediction': predicted_class
        }
        patients.setdefault(session['username'], []).append(patient_info)

        # No flash here, directly render result page
        return render_template('result.html',
                               patient_name=patient_name,
                               prediction=predicted_class,
                               image_file=filename)

    return render_template('upload.html')

@app.route('/patients')
def patient_list():
    if 'username' not in session:
        flash('Please login first.', 'warning')
        return redirect(url_for('login'))
    user_patients = patients.get(session['username'], [])
    return render_template('patients.html', patients=user_patients)

if __name__ == '__main__':
    app.run(debug=True)
