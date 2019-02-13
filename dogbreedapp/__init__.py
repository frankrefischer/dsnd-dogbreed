from flask import Flask
import os
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.split(__file__)[0], 'uploads')

from dogbreedapp import routes

