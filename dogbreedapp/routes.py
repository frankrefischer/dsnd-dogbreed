from dogbreedapp import app

from flask import render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
from dogbreedapp.classification import classify

def upload_folder():
    "utility to get the configured upload folder in a nicer, more readable way."
    return app.config['UPLOAD_FOLDER']

def upload_filepath(filename):
    "utility to get the upload filepath for a filename in a nicer, more readable way."
    return os.path.join(upload_folder(), filename)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload an image file.
    """

    file = request.files['query']
    filepath = upload_filepath(secure_filename(file.filename))
    file.save(filepath)
    classification = classify(filepath)
    classification['filename'] = file.filename
    return render_template('index.html', classification=classification)

@app.route('/uploads/<path:filename>')
def download_file(filename):
    """
    Serve the uploaded images.
    """
    return send_from_directory('uploads', filename, as_attachment=True)
