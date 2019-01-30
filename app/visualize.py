import plotly
import plotly.graph_objs as go
import pandas as pd
import argparse
import os
import shutil
import sys, traceback

MAPBOX_ACCESS_TOKEN = "YOUR TOKEN HERE"

def plot_main(all_records,mapbox_access_token,output_location):
    all_df = pd.read_csv(all_records)
    colors = ['green' if x < 5 else 'orange' if x < 10 else 'red' for x in all_df['num_vehicles'].values]
    html_string = [
    '<a href="'+str(x)+'.html" ' for x in all_df['CameraID']
    ]
    data = [
        go.Scattermapbox(
            lat= all_df['Latitude'],
            lon= all_df['Longitude'],
            mode='markers',
            marker=dict(
                color=colors,
                size=7
            ),
            hoverinfo = 'text',
            hoverlabel = dict(
                bgcolor="lightgrey",
                font=dict(color=colors)
            ),
            text=[str(int(all_df['num_vehicles'].values[x]))+" vehicles detected<br>"+
                html_string[x]
                  +""" style="color:black">{}</a>""".format(
                    "Camera "+str(all_df['CameraID'].values[x]))
                 for x in range(len(all_df))]
        )
    ]

    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=1.353641,
                lon=103.818260
            ),
            pitch=0,
            zoom=10
        ),
    )
    plotly.offline.plot({
        "data": data,
        "layout": layout,
    },auto_open=False,filename=output_location)
    return None

def plot_single(single_record,output_location):
    single_df = pd.read_csv(single_record)
    maxlength = min(len(single_df),200)
    data = [
        go.Scatter(
            x=single_df['datetime'][-maxlength+1:],
            y=single_df['num_vehicles'][-maxlength+1:]
        )
    ]
    plotly.offline.plot({
        "data": data,
    },auto_open=False,filename=output_location)
    return None

html_text = ["""{% extends "base.html" %} {% block content %} \
<div><div style ="width: 50%; float: left;">\
<h2>Camera Image</h2>\
<img src=""",""" width="80%" height="50%">\
</div>\
<div style ="width: 50%; float: right;">\
<h2>Detection History</h2>\
<iframe frameborder="0" seamless="seamless" width="80%" height="50%" src=""",""" ></iframe> \
</div></div>{% endblock %}"""]
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-record_folder','--record_folder')
  parser.add_argument('-all_records','--all_records')
  args = parser.parse_args()

  if not os.path.isfile(args.all_records):
      raise Exception("cannot find all records")
  try:
    plot_main(args.all_records,MAPBOX_ACCESS_TOKEN,'./templates/main_plot.html')
  except Exception as e:
      print(e)
      traceback.print_exc(file=sys.stdout)
      pass
  if not os.path.isdir(args.record_folder):
      raise Exception("cannot find record folder")
  record_list = os.listdir(args.record_folder)
  for single_record in record_list:
      # print(single_record)
      try:
          plot_single(os.path.join(args.record_folder,single_record),
          './templates/'+single_record.split('.')[0]+'_graph.html')
          html_file = open(os.path.join('./templates',single_record.split('.')[0]+".html"),"w")
          html_file.write(html_text[0]+"'/static/"+single_record.split('.')[0]+".jpg"+"'"+html_text[1]+
          "'"+single_record.split('.')[0]+"_graph.html"+"'"+html_text[2])
          html_file.close()
          shutil.copyfile(os.path.join('./output_image',single_record.split('.')[0]+".jpg"),
          os.path.join('./static',single_record.split('.')[0]+".jpg"))
      except Exception as e:
          print(single_record)
          print(e)
          traceback.print_exc(file=sys.stdout)
          pass
