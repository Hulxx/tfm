# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:56:27 2020

@author: Usuario
"""
from __future__ import absolute_import
import argparse
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io.gcp.bigquery import parse_table_schema_from_json

import json
import pandas as pd
import random
import time
import base64
#import librosa
import numpy as np

from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Model():
    def __init__(self):
        model_dir = 'D:/MDA/TFM/gft/modelsaved.h5'
        model = tf.keras.models.load_model(model_dir)
        model = model.compile()
        print('Model loaded')


class preprocessAudio(beam.DoFn):
    def process(self, element):
        
        start = time.time()
        lon = random.uniform(36.6254,43.26271)
        lat = random.uniform(-6.97061,-0.37739)
        geo = "{},{}".format(lat,lon)
        item = json.loads(element)
        key = vars.json['params'][1]
        payload = key.get('soundFileContent')
        wav_file = open("wave.wav", "wb")
        decode_string = base64.b64decode(payload)
        wav_file.write(decode_string)
        example_audio_path = (model_dir)
        feature = []
        
        def parser(file_path):
            # Here kaiser_fast is a technique used for faster extraction
            X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            # We extract mfcc feature from data
            mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            feature.append(mels)
            
            return [feature]
        
        temp = parser(example_audio_path)
        temp = np.array(temp)
        data = temp.transpose()
        X = np.empty([1, 128])
        X= X.reshape(1, 16, 8, 1)
        end= time.time()
        elaspedtime = end - start
        return [(item["jsonrpc"],
                 item["id"],
                 item["method"],
                 item["params"][0]["userid"],
                 item["params"][1]["soundFileContent"],
                 item["params"][2]["filename"],
                 geo,
                 elaspedtime)]

class audioClassifier(beam.DoFn):
    model = Model()
    print('document classifier')
    def process(self, element):
        item = json.loads(element)
        pred = model.predict(X)[0]
        Results= ['0.Air Conditioner', pred[0]],['1.car_horn', pred[1]],['2.children_playing', pred[2]],['3.dog_bark', pred[3]],['4.drilling', pred[4]],['5.engine_idling',pred[5]],['6.gun_shot',pred[6]],['7.jackhammer',pred[7]],['8.siren',pred[8]],['9.street_music',pred[9]]
        predicted_class_indices=np.argmax(pred,axis=1)
        
        return [(item["params"][2]["filename"],
                geo,
                elaspedtime,
                predicted_class_indices)]

def run(argv=None, save_main_session=True):
    parser = argparse.ArgumentParser()
    
    #1 Replace your hackathon-edem with your project id
    parser.add_argument('--input_topic',
                      dest='input_topic',
                      default='projects/gft-app-294621/topics/audio_topic',
                      help='Input file to process.')
    
    #2 Replace your hackathon-edem with your project id
    parser.add_argument('--input_subscription',
                      dest='input_subscription',
                      default='projects/gft-app-294621/subscriptions/audio_sub',
                      help='Input Subscription')
    
    parser.add_argument('--output_table', 
                        required=True,
                        dest='gft-app-294621:audios_dataset.classified_data',
                        help=
                        ('Output BigQuery table for results specified as: PROJECT:DATASET.TABLE '
                        'or DATASET.TABLE.'))
    
    known_args, pipeline_args = parser.parse_known_args(argv)
    
    pipeline_options = PipelineOptions(pipeline_args)
    
    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    
    google_cloud_options.project = 'gft-app-294621'
    google_cloud_options.job_name = 'gftapp'
    # Uncomment below and add your bucket if you want to execute on Dataflow
    google_cloud_options.staging_location = 'gs://audio_app/binaries'
    google_cloud_options.temp_location = 'gs://audio_app/temp'
    google_cloud_options.region ='europe-west1'
    
    pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
    pipeline_options.view_as(StandardOptions).streaming = True
    
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    
    p = beam.Pipeline(options=pipeline_options)
    
    data = (p | 'Read from PubSub' >> beam.io.ReadFromPubSub(subscription=known_args.input_subscription))
    
    processed_data = (data | beam.ParDo(preprocessAudio()))
    
    classified_data = (processed_data | beam.ParDo(audioClassifier()))
    
    table_schema = parse_table_schema_from_json(json.dumps(json.load(open("schema.json"))["schema"]))
    
    classified_data |   'Write to Big Query' >> beam.io.WriteToBigQuery(args.output, schema=table_schema, create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
    
    result = p.run()
    result.wait_until_finish()

if __name__ == '__main__' :
    run()
