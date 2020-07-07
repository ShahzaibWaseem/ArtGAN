import os
from glob import glob
from flask import Flask, render_template

app = Flask(__name__)

STYLE_TRANSFER_PATH=os.path.join("static", "images", "NeuralStyleTransfer", "*.jpg")
ART_GENERATION_PATH=os.path.join("static", "images")

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/contact")
def contact():
	return render_template("contact.html")

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/styleTransfer")
def styleTransfer():
	listOfFiles=[]
	for imageFilePath in glob(STYLE_TRANSFER_PATH):
		listOfFiles.append(os.path.join("..", imageFilePath))
	listOfFiles=sorted(listOfFiles)
	return render_template("styleTransfer.html", listOfFiles=listOfFiles)

@app.route("/williamMorris")
def williamMorris():
	listOfFiles=[]
	for imageFilePath in glob(os.path.join(ART_GENERATION_PATH, "william-morris-w", "*.jpg")):
		listOfFiles.append(os.path.join("..", imageFilePath))
	listOfFiles=sorted(listOfFiles)
	return render_template("artGeneration.html", listOfFiles=listOfFiles)

@app.route("/polarPlots")
def polarPlots():
	listOfFiles=[]
	for imageFilePath in glob(os.path.join(ART_GENERATION_PATH, "PolarPlots", "*.png")):
		listOfFiles.append(os.path.join("..", imageFilePath))
	listOfFiles=sorted(listOfFiles)
	return render_template("artGeneration.html", listOfFiles=listOfFiles)

@app.route("/fashionMNIST")
def fashionMNIST():
	listOfFiles=[]
	for imageFilePath in glob(os.path.join(ART_GENERATION_PATH, "FashionMNIST", "*.png")):
		listOfFiles.append(os.path.join("..", imageFilePath))
	listOfFiles=sorted(listOfFiles)
	return render_template("artGeneration.html", listOfFiles=listOfFiles)