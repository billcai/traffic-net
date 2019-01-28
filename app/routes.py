from flask import render_template, flash, redirect, url_for, request, send_from_directory
from app import app
import pandas as pd
import os

app.static_url_path = '/output_image'
app.static_folder = 'output_image'

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')
@app.route('/main_plot')
def main_plot():
    return send_from_directory('templates','main_plot.html')

@app.route('/<CameraID>')
def image_page(CameraID):
    return render_template(str(CameraID),
    image_location=str(CameraID).split('.')[0]+'.jpg',
    graph_location=str(CameraID).split('.')[0]+'_graph.html')
@app.route('/<CameraID>_graph')
def image_graph(CameraID):
    return send_from_directory('templates',str(CameraID)+'_graph.html')
