import os
from flask import Flask, render_template,request
#print("Code executed successfully FLASK")
import librosa
import numpy as np
import tensorflow as tf
from model_weight import model_weights,configuration
import json
import h5py


app = Flask(__name__)

#model = tf.keras.models.load_model("./model-001-0.815760.h5")
#model = tf.keras.models.load_model('./model-001-0.541899.h5')
#model = tf.keras.models.load_model('./model-002-0.586592.h5')
FileName = './model-001-0.541899.h5'
#FileName = './model-002-0.586592.h5'

#FileName = './model-007-0.413408.h5'
#FileName = './model-001-0.815760.h5'
@app.route("/")
def hello():
	print("Hello function execting")
	return render_template("index.html")

@app.route("/sub",methods  =["POST"])
def submit():
	#max_length = 300
	if request.method == "POST":
		bias,kernel = model_weights(FileName)
		config = configuration(FileName)
		audio_path = request.form["myfile"]
		#print(audio_path)
		x , sr = librosa.load(audio_path)	
		chroma_cens = librosa.feature.chroma_cens(x,sr=sr)
		cens = chroma_cens.mean(axis=0)
		cens = cens.reshape(1,cens.shape[0])
		rmse = librosa.feature.rms(x)

		spectral_s = librosa.feature.spectral_centroid(x)
		spectral_b = librosa.feature.spectral_bandwidth(x)
		roll_off = librosa.feature.spectral_rolloff(x,sr=sr)
		zerocross = librosa.feature.zero_crossing_rate(x)
		mfccs = librosa.feature.mfcc(x, sr=sr)
		#Input_Sample = np.concatenate((cens,rmse,spectral_s,spectral_b,roll_off,zerocross,mfccs),axis=0)
		
		Input_Sample = np.mean(np.concatenate((cens,rmse,spectral_s,spectral_b,roll_off,zerocross,mfccs),axis=0),axis=1)
		Input_Sample = np.reshape(Input_Sample,(1,Input_Sample.shape[0]))
		#pred = np.argmax(model.predict(Input_Sample.T),axis=1)
		pred = np.argmax(forward_propagation(Input_Sample,kernel,bias,config))
		#zero = np.count_nonzero(pred == 0)
		#one  = np.count_nonzero(pred == 1)
		#Two = np.count_nonzero(pred == 2)
		#State = np.argmax(np.array([zero,one,Two]))
		print("Predcited Class is ",pred)
		State = "Donno"
		State = str(pred)
		if State == '0':
			State = "Abnormal Hen need to be diagnosed"
		elif State == '1':
			State = "Normal Hen, Please Dont disturb it"
		elif State == '2':
			State = "Unknown State Manual checking is required"
	return render_template("sub.html",statement = State)
	#return render_template("sub.html",statement = State)
def linear_reg(Input,Weight,Bias):
  return np.dot(Input,Weight)+Bias
def relu(Input):
  return np.maximum(Input,0)
def sigmoid(Input):
  return 1.0/(1 + np.exp(-Input))
def softmax(Input):
  return np.exp(Input) / np.sum(np.exp(Input), axis=1)
  
def forward_propagation(Input_Sample,kernel,bias,config):
  lin = Input_Sample
  for layer in kernel:
    #print(layer)
    lin = linear_reg(lin,kernel[layer],bias[layer])
    if config[layer] == 'relu':
      lin = relu(lin)
    if config[layer] == 'softmax':
      lin = softmax(lin)
  return lin
if __name__ == "__main__":
	#os.environ['FLASK_ENV'] = 'development' 
	app.run(debug=False)
