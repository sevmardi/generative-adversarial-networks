import tensorflow as tf
import numpy as np



def load_vgg(sess, vgg_path):
	"""
	Load the model 
	"""
	# Load the model and weights
	model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

	#Get the tensors to be retured from graph
	graph = tf.get_default_graph()

	image_input= graph.get_tensor_by_name('image_input:0')
	keep_prob = graph.get_tensor_by_name('keep_prob:0')
	layer3= graph.get_tensor_by_name('layer3_out:0')
	layer4 = graph.get_tensor_by_name('layer4_out:0')
	layer7 = graph.get_tensor_by_name('layer7_out:0')

	return image_input, keep_prob, layer3, layer4, layer7

