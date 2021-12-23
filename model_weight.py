import h5py
import json

def model_weights(model):
  f1 = h5py.File(model,"r+")
  model,optimizer = list(f1)
  layers = list(f1[model])
  layers = [layer for layer in layers if "dense" in layer]
#  print("Layers",layers)
  #sub_layer = list(f1[model][layers[0]])
  #print(sub_layer)
  bias = {}
  kernel = {}
  for i in range(len(layers)):
  #  print(f1[model][layers[i]][layers[i]]['bias:0'][:])
    bias[layers[i]]=f1[model][layers[i]][layers[i]]['bias:0'][:]
    kernel[layers[i]] = f1[model][layers[i]][layers[i]]['kernel:0'][:]
#  bias = np.array(bias)
#  kernel = np.array(kernel)
  return (bias,kernel)


def configuration(model):
  f1 = h5py.File(model,"r+")
  model_configuration = f1.attrs.get('model_config')
  #print(model_configuration)
  model_configuration = json.loads(model_configuration)
  activation = {}
  for i in range(len(model_configuration["config"]["layers"])):
    if model_configuration["config"]["layers"][i]["class_name"] == "Dense":
      activation[model_configuration["config"]["layers"][i]['config']['name']] = model_configuration["config"]["layers"][i]['config']['activation']
  return activation