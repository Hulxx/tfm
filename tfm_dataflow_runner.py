#!/usr/bin/python3

from __future__ import absolute_import
import argparse
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import SetupOptions

import json
import pandas as pd
import random
import time
import base64
import librosa
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
        model_dir = '/home/carlesredon/tfm/mymodel2.h5'
        model = tf.keras.models.load_model(model_dir)
        model = model.compile()
        print('Model loaded')

# Audio pre-processing, todo el preprocesamiento
class preprocessAudio(beam.DoFn):
    def process(self, element):
        start = time.time()
        lon = random.uniform(-60,60)
        lat = random.uniform(-60,60)
        location = "{},{}".format(lat,lon)
        item = json.loads(element)
        key = vars_json['params'][1]
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
        return [{'jonrpc' : item["jsonrpc"],
                     'id' : item["id"],
                     'method' : item["method"],
                     'userid': item["params"][0]["userid"],
                     'soundFileContent': item["params"][1]["soundFileContent"],
                     'filename' : item["params"][2]["filename"],
                     'geo' : location,
                     'elapsedtime' : elaspedtime}]
	
# Model prediction 
class audioClassifier(beam.DoFn):
    model = Model()
    print('document classifier')
    def process(self, element):
        item = json.loads(element)
        df = pd.DataFrame({item["params"][2]["filename"],'geo' : location,'elapsedtime' : elaspedtime}, index[0])
        pred = model.predict(X)[0]
        Results= ['0.Air Conditioner', pred[0]],['1.car_horn', pred[1]],['2.children_playing', pred[2]],['3.dog_bark', pred[3]],['4.drilling', pred[4]],['5.engine_idling',pred[5]],['6.gun_shot',pred[6]],['7.jackhammer',pred[7]],['8.siren',pred[8]],['9.street_music',pred[9]]
        predicted_class_indices=np.argmax(pred,axis=1)
        
        return [{'filename' : item["params"][2]["filename"],
                     'geo' : location,
                     'elapsedtime' : elaspedtime,
                     'prediction' : predicted_class_indices}]
    

# Crear index to bigquery
# class indexAudio(beam.DoFn):
#     def run(argv=None):
        
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_mode',
#                         default='stream',
#                         help='Streaming input or file based batch input')
#     parser.add_argument('--input_topic',
#                         default='projects/tfmfinal/topics/audio',
#                         required=True,
#                         help='Topic to pull data from.')
#     parser.add_argument('--output_table', 
#                         required=True,
#                         help=
#                         ('Output BigQuery table for results specified as: PROJECT:DATASET.TABLE '
#                         'or DATASET.TABLE.'))
#     known_args, pipeline_args = parser.parse_known_args(argv)

#     pipeline_options = PipelineOptions(pipeline_args)
#     pipeline_options.view_as(SetupOptions).save_main_session = True

#     if known_args.input_mode == 'stream':
#         pipeline_options.view_as(StandardOptions).streaming = True

#     with beam.Pipeline(options=pipeline_options) as p:

#         price = ( p
#                 | 'ReadInput' >> beam.io.ReadFromPubSub(topic=known_args.input_topic).with_output_types(six.binary_type))
#                 | 'Decode'  >> beam.Map(decode_message)
#                 | 'Parse'   >> beam.Map(parse_json) 
#                 | 'Write to Table' >> beam.io.WriteToBigQuery(
#                         known_args.output_table,
#                         schema='jonrpc' : item["jsonrpc"],
#                         'id' : item["id"],
#                         'method' : item["method"],
#                         'userid': item["params"][0]["userid"],
#                         'soundFileContent': item["params"][1]["soundFileContent"],
#                         'filename' : item["params"][2]["filename"],
#                         'geo' : location,
#                         'elapsedtime' : elaspedtime,
#                         'prediction' : prediction,
#                         write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))

def run(argv=None, save_main_session=True):
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_topic', dest='projects/tfmfinal/topics/audio', default='poner mi topic', help='Input file to process')

	parser.add_argument('--input_subscription', dest='input_subscription', default = 'projects/tfmfinal/subscriptions/audio_sub', help = 'Input Subscription')
	known_args, pipeline_args = parser.parse_known_args(argv)
	pipeline_options = PipelineOptions(pipeline_args)
	google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
	google_cloud_options.project = 'tfmfinal'
	google_cloud_options.job_name = 'tfmjob'
    
    pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
    pipeline_options.view_as(StandardOptions).streaming = Trueproject='tfmfinal',
    pipeline_options.view_as(StandardOptions).job_name='my_job',
    pipeline_options.view_as(StandardOptions).temp_location='gs://audio/temp',
    pipeline_options.view_as(StandardOptions).region='us-central1',
    pipeline_options.view_as(StandardOptions).output_table="tfmfinal:tfm_bigquery.tabla_tfm")

	pipeline_options.view_as(SetupOptions).save_main_session =  save_main_session

	p = beam.Pipeline(options = pipeline_options)

	data = p | beam.io.ReadFromPubSub(topic=known_args.input_topic)

	#data | 'Print Quote' >> beam.Map(print)
	processed_data = data | beam.ParDo(preprocessAudio())


	classified_data = processed_data | beam.ParDo(audioClassifier())


	classified_data | 'Audio classified' >> beam.io.WriteToBigQuery(known_args.output_table,schema='jonrpc' : item["jsonrpc"],
                        'id' : item["id"],
                        'method' : item["method"],
                        'userid': item["params"][0]["userid"],
                        'soundFileContent': item["params"][1]["soundFileContent"],
                        'filename' : item["params"][2]["filename"],
                        'geo' : location,
                        'elapsedtime' : elaspedtime,
                        'prediction' : prediction,
                        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)))

	result = p.run()
	result.wait_until_finish()

if __name__ == '__main__':
	logging.getLogger().setLevel(logging.INFO)
	run()



