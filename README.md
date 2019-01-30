# Traffic-Net
A four evening Tensorflow and Flask app to detect vehicles on live LTA cameras using deep learning-based object detection. Goals: Lightweight, lightning fast and likeable visuals.

## What is this about?
A barebones Tensorflow and Flask app that you can deploy! The app downloads images from cameras available on LTA's DataMall, runs fast and accurate a Tensorflow implementation of a trained SSD-Mobilenet object detection model that finds vehicles, and visualizes the results on a main map:

![alt-text](https://github.com/billcai/traffic-net/blob/master/images/index_page.png "Main Page")

For each camera, the most recent detection results are also shown, along with a chart tracking the last few detections.

![alt-text](https://github.com/billcai/traffic-net/blob/master/images/single_page.png "Single Camera Page")

## What do I need to run this?

Preferably, you need a Linux or Mac machine (Windows is perfectly fine, but you have to find the equivalent instructions). First, set up a virtualenv:

```
cd traffic-net
python -m venv traffic-net
source traffic-net/bin/activate
```
Install required packages by using:
```
pip install -r requirements.txt
```
You need to signup for a Mapbox  account to get a Mapbox Access Token to display maps, and also you need to obtain a LTA DataMall token. Both are free to get, and you need to input the former in `app/visualize.py` and the latter in `app/download_image.py`. Then, use a screen (like tmux) and run the following command in a screen to run flask.
```
export FLASK_APP=traffic-net_flask.py
flask run
```
Then, run the following to start scraping and detecting!
```
cd app
python temp_scraper.py
```
And use a separate screen to start visualizing:
```
cd app
python visualize_loop.py
```
And head over to `localhost:5000` to see live traffic conditions in Singapore

## Credits

For training the SSD object detection model and visualizing the results, I used [Tensorflow's Object Detection research repository](https://github.com/tensorflow/models/tree/master/research/object_detection). [Flask](http://flask.pocoo.org/) and [Plotly](https://github.com/plotly/plotly.py) were used as the web framework and visualization library respectively.
