from flask import Flask

app = Flask(__name__)

from dogbreedapp import routes

