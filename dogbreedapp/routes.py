from dogbreedapp import app

from flask import render_template, redirect

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload an image file
    """
    return render_template('index.html', contains_human_face=False,
                                contains_dog=True,
                                dog_breed='mambo jambo')