import json
import requests
import pandas as pd
import urllib.request
import os
import argparse

API_KEY = "<Your LTA Datamall API Key here>"

def get_response(api_key):
    url = 'http://datamall2.mytransport.sg/'
    service = 'ltaodataservice/Traffic-Images?'
    headers = {'AccountKey':api_key,'accept':'application/json'}
    response = requests.get(url+service,headers=headers)
    json_response = json.loads(response.text)
    return json_response

def response_to_csv(json_response,csv_filename):
    latitude_list = []
    longitude_list = []
    camera_id_list = []
    image_location_list = []
    if 'value' not in json_response:
        raise Exception("Cannot find value key in json response")
    for indiv_camera in json_response['value']:
        if (('Latitude' in indiv_camera) and
            ('Longitude' in indiv_camera) and
            ('CameraID' in indiv_camera) and
            ('ImageLink' in indiv_camera)):
            latitude_list.append(indiv_camera['Latitude'])
            longitude_list.append(indiv_camera['Longitude'])
            camera_id_list.append(int(indiv_camera['CameraID']))
            image_location_list.append(indiv_camera['ImageLink'])
    dataframe = pd.DataFrame({
        'Latitude':latitude_list,
        'Longitude':longitude_list,
        'CameraID':camera_id_list,
        'ImageLink':image_location_list
    })
    if os.path.isfile(csv_filename):
        old_df = pd.read_csv(csv_filename)
        if 'num_vehicles' in list(old_df):
            dataframe['num_vehicles'] = [0 for i in range(len(dataframe))]
            for single_id in dataframe['CameraID'].values:
                if single_id in list(old_df['CameraID']):
                    num_vehicles = old_df.loc[old_df['CameraID']==single_id,'num_vehicles']
                    dataframe.loc[dataframe['CameraID']==single_id,'num_vehicles'] = int(num_vehicles)
    dataframe.to_csv(csv_filename,index=False)
    return dataframe

def df_to_images(dataframe,output_folder):
    for i in range(len(dataframe)):
        urllib.request.urlretrieve(
            dataframe['ImageLink'].values[i],
            os.path.join(output_folder,str(dataframe['CameraID'].values[i])+'.jpg')
        )
    return None

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-image_folder','--image_folder',help="Image Folder")
  parser.add_argument('-csv_filename','--csv_filename',help ="CSV Filename")
  args = parser.parse_args()

  json_response = get_response(API_KEY)
  dataframe = response_to_csv(json_response,args.csv_filename)
  df_to_images(dataframe,args.image_folder)
