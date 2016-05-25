#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
ID: convNet_Ex02
Description:
	The convolutional Neural Network base on LeNet tutorial on DeepLearning.com, from which multiple theano APIs are invoked to fast construt and reduce redundant works.
	Two convolutional & max poooling layer + logistic regression output layer
	About methods' return code: in general, 1 stands for success, otherwise ,which indicates error type,failure;
	To be tested on MNIST
Interface:
	1. input: the training data , which has already been rendered to proper size; [the inintializing value for filters]; training parameters; trival input
	2. output: minimized error; output of logistic regression layer
Date:
	5.18.2016
Credits:
	all credits to the authors of the tutorial on http://deeplearning.net
'''
__author__ = "Ruogu Gao"

import os
import sys
import timeit
import logging

import numpy
import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer
from dataLoader import dataLoader

logging.basicConfig( filename = 'convNet.log', level = logging.DEBUG )
'''
Phase_II_tips:
	1. reload the constructor
	2. multiple type and shape check point
	3. pickle( serialization ) the trained network for faster prediction
'''

class ConvNet(object):
	'''
	the prototype convNet, convNet_Ex02
	'''
	def __init__(self, randomSeed, 
		filter_shape_1, image_shape_1, pool_size_1, 
		filter_shape_2, image_shape_2, pool_size_2,
		hiddenLayer_input_d,
		hiddenlayer_output_d,
		n_class ):
		'''
		construct the basic structure of ConvNet: two convPooling layer, one normal hiddenlayer and a logistic
		refer the LeNetConvPoolLayer for the input parameter meanings
		NOTE	
			1.the input should be correctly transformed as 4D tensor or unpredictable behaviour might occur
			2.the input "filter_shape" implicitly determined the feature panel number in one LeNetConvPoolLayer
			3.the downsample panel number is equal to it's feature panel and it's panel size are determined by pool_size
		'''
		##phase_II: add some type check maybe


		##record all input parameters
		##to this stage, format of all input parameter should be adjusted properly
		self._fshape1 = filter_shape_1
		self._fshape2 = filter_shape_2
		self._poolS1 = pool_size_1
		self._poolS2 = pool_size_2
		self._imageS1 = image_shape_1
		self._imageS2 = image_shape_2
		self._classN = n_class
		self._hiddenInputN = hiddenLayer_input_d
		self._hiddenUnitN = hiddenlayer_output_d

		self.symbolic_x = T.dmatrix('symbolic_x')
		self.symbolic_y = T.lvector('symbolic_y')
		self._incoming = self.symbolic_x.reshape( self._imageS1)
		'''
		self._test_set_x = 
		self._test_set_y = 
		self._train_data_x = 
		self._train_data_y =
		self._validate_data_x = 
		self._validate_data_y = 

		learning_rate=0.1, n_epochs=200,
                     dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500
		'''
		self._test_data_ready = False
		self._validate_data_ready = False
		self._train_data_ready = False
		##set up the structure 
		#rng, input, filter_shape, image_shape, poolsize=(2, 2)):
		self._layer_conv1 = LeNetConvPoolLayer( randomSeed, self._incoming, 
			self._fshape1, self._imageS1, self._poolS1 )

		##Phase_II: check the imageS2 by imageS1 and poolS1

		self._layer_conv2 = LeNetConvPoolLayer( randomSeed, self._layer_conv1.output, 
			self._fshape2, self._imageS2, self._poolS2 )


		self._hidden_input = self._layer_conv2.output.flatten( 2 )

		self._layer_hidden1 =  HiddenLayer( randomSeed, self._hidden_input, 
			n_in = self._hiddenInputN,
			n_out = self._hiddenUnitN,
			activation = T.tanh )

		self._layer_logistic = LogisticRegression( self._layer_hidden1.output, 
			n_in = self._hiddenUnitN,
			n_out = self._classN )

		self.cost = self._layer_logistic.negative_log_likelihood( self.symbolic_y )

		self.allParameter = self._layer_logistic.params + self._layer_hidden1.params + self._layer_conv2.params + self._layer_conv1.params

		self.gradients = T.grad( self.cost, self.allParameter )

		print("... Model construction complete.\n")


	def  trainingStart(self, learning_rate, epochN, batch_size ):#################################################################################
		'''
		Start the iterative training process
		Early-Stopping technique implemented
		'''
		#check all parameter needed is ready
		if self._DataCheck() == False :
			print("the one of the training, ")
			return 0

		self._learningRate = learning_rate
		self._batchSize = batch_size
		self._epochN = epochN
		#set up update queue
		self._updates = [ ( param_i ,  param_i - self._learningRate*grad_i) 
			for param_i, grad_i in zip(  self.allParameter, self.gradients)
		]
		#setup test, validation, training symbolic functions
		index = T.lscalar()

		self.test_model = theano.function(
			[ index ],
			self._layer_logistic.errors( self.symbolic_y ),
			givens = {
				self.symbolic_x : self._test_set_x[ index * self._batchSize : (index+1) * self._batchSize ],
				self.symbolic_y : self._test_set_y[ index * self._batchSize : (index+1) * self._batchSize ]
			}
		)

		self.validate_model = theano.function(
			[ index ],
			self._layer_logistic.errors( self.symbolic_y ),
			givens = {
				self.symbolic_x : self._validate_data_x[ index*self._batchSize : (index+1)*self._batchSize ],
				self.symbolic_y : self._validate_data_y[ index*self._batchSize : (index+1)*self._batchSize ]
			}
		)

		self.train_model = theano.function(
			[ index ],
			self.cost,
			updates = self._updates,
			givens = {
				self.symbolic_x : self._train_data_x[ index*self._batchSize : (index+1)*self._batchSize ],
				self.symbolic_y : self._train_data_y[ index*self._batchSize : (index+1)*self._batchSize ]	
			}
		)
		print( "... Training modules all set, starting training process. \n" )
		##early-stopping implemented through so-called "patience" parameter
		'''
		self._learningRate = learning_rate
		self._batchSize = batch_size
		self._epochN - epochN
		'''
		#basic training parameters
		self._patience = 10000;# run at least these many iterations, every iteration train on "_batchSize" numnber of data
		self._patience_increase = 2
		self._improvement_threshold = 0.995

		self._total_train_batchN = (self._train_data_x.get_value(borrow=True).shape[0]) // self._batchSize
		self._total_test_batchN = (self._test_set_x.get_value(borrow=True).shape[0]) // self._batchSize
		self._total_validate_batchN = (self._validate_data_x.get_value(borrow=True).shape[0]) // self._batchSize

		self._validate_frequence  = min( self._total_train_batchN , self._patience // 2 )
		logging.info( "_total_train_batchN: " + str( self._total_train_batchN ) )
		logging.info( "_validate_frequence: " + str( self._validate_frequence ) )

		self._best_validate_loss_mean = numpy.inf
		self._test_score = 0
		self._current_validate_loss_mean =0
		self._current_cost = 0
		self._current_epoch = 0
		self._done_flag = False
		self._iterN = 0 #thus it starts from 0

		self._local_minimum = []

		## traing start
		#one step training->periodic validate->evaluate through test data->iteration untill training data or patience run outs
		while ( self._current_epoch < self._epochN ) and ( self._done_flag != True ):
			self._current_epoch += 1
			# every traning epoch, we go through ALL given training data cases, which are devided into "_total_train_batchN" groups(batches),and "_batchSize" cases each group.
			#NOTE: every single train step( ie. one invoke of method "train_model" ) walk through "_batchSize" number of training cases
			print( "Entering Epoch: %d, at iteration: %d" %( self._current_epoch, self._iterN))																																																																
			for mini_batch_index in range( self._total_train_batchN ):
				self._iterN = ( self._current_epoch - 1 ) * self._total_train_batchN + mini_batch_index

				self._current_cost = self.train_model( mini_batch_index )#envoloped the iteration and parameter update

				#check whether to validate at this iteration( determined by _validate_frequence )
				if ( ( self._iterN + 1 )% self._validate_frequence == 0 ):
					#run validation by current model parameters on ALL validation cases
					self._current_validate_loss = [ self.validate_model(i) for i in range( self._total_validate_batchN )]
					self._current_validate_loss_mean = numpy.mean( self._current_validate_loss )

					#check whether the validation loss has improved
					if ( self._current_validate_loss_mean <  self._best_validate_loss_mean):
						#note:should there be any improve on loss mean, renew '_best_validate_loss_mean' and if the improvement happens to exceed the given threshold( _improvement_threshold ), we extend the patience 
						#after that, we calculate the test_score( -log likelihood ) on best validation parameter
						if ( self._current_validate_loss_mean <= self._best_validate_loss_mean * self._improvement_threshold ):
							#extend the patience
							self._patience = max( self._patience,  self._iterN * self._patience_increase)
						##update "_best_validate_loss_mean"
						self._best_validate_loss_mean = self._current_validate_loss_mean

						##the corresponding test_score on ALL test case
						test_score_tmp = [ self.test_model(j) for j in range( self._total_test_batchN ) ]
						self._test_score = numpy.mean( test_score_tmp )
						print( "Test score at iter %d is %f %%" % ( self._iterN, self._test_score*100.0 )  )
						#Ex01 added
						self._local_minimum.append( ( self._iterN, self._test_score) )

				##check whether the patience's been "run out"
				if ( self._patience < self._iterN ):
					self._done_flag = True
					break
		print("Training complete and ready to roll.\n")
		print ("the current best performance on MNIST is %f" % ( self._test_score, ))
		return 1

	def feed(self, pilgrim):
		'''
		get input and return the output of regression layer
		make sure it's called after the training
		and check the input format/type
		'''
		prophet_pred = theano.function(
			input = self.symbolic_x,
			output = self._layer_logistic.y_pred
			)

		return  prophet_pred( pilgrim )


	##all those getter and setters
	def setData(self, imageSize, imageN_test, imageN_train, imageN_validate, 
						testPath, trainPath, validatePath, testRange, trainRange, 
							validateRange, testY, trainY, validateY):

		loader = dataLoader( imageSize, imageN_test, imageN_train, imageN_validate )
		loader.setData( testPath, trainPath, validatePath )
		self._datasets = loader.loadData(  testRange, trainRange, validateRange, testY, trainY, validateY )
		self._setTrainingData( self._datasets[1] )
		self._setValidateData( self._datasets[2] )
		self._setTestData( self._datasets[0] )

		print( "... Triple DataSet in position. \n" )


	def _setTrainingData(self, incoming ):##################################
		##Phase_II: check type input should be a fmatrix(theano)

		self._train_data_x = incoming[0]
		self._train_data_y = incoming[1]
		self._train_data_ready = True

	def _setValidateData(self, incoming ):##################################
		##Phase_II: check type

		self._validate_data_x = incoming[0]
		self._validate_data_y = incoming[1]
		self._validate_data_ready = True

	def _setTestData(self, incoming ):######################################
		##Phase_II: chech type

		self._test_set_x = incoming[0]
		self._test_set_y = incoming[1]
		self._test_data_ready = True
		

	def _DataCheck(self):
		if (self._test_data_ready == True) and (self._train_data_ready == True) \
			and (self._validate_data_ready == True) :
			return True
		else:
			return False
			

	def writeTrainingLocalMinimum(self, fileName):
		##write all recorded local minimums(and it's iteration number) to the specified file
		##this format could be easily plotted by gnuplot
		with open( fileName, 'w' ) as file:
			for tmp in self._local_minimum:
				file.write( str(tmp[0]) + ' ' + str(tmp[1]) )
				file.write( '\n' )

		print( "... All local minimum has been write to %s" % ( fileName, ))


	def initFirstLayerFilterWeights(self, ):###Phase_II##############################################################
		'''
		inintialize the filter weights of the FIRST convPooling Layer filter weights
		return 1 when succeed
		!!!some mechanism should be deployed to ensure this method be called BEFORE the 'trainingStart', maybe
		before any other methods save '__init__'
		'''
		if weight_ini_flag == true:
			return 0
		###under construction

	def viewStructure(self):####################################################################################
		##print the structure information
		pass


	def get(self):#############################################################################################
		# may try the '命名关键字参数'' feature
		pass



##---------------------------------------------------------------Nirn_Plane-------------------------------------------------------------------------------------
#the testing sequence on Complex I thermoXXX
#set random seed 
rSeed = numpy.random.RandomState( 33466 )
instanceEx01 = ConvNet( randomSeed = rSeed, 
		filter_shape_1 = ( 20, 1, 5, 5 ), image_shape_1 = ( 10, 1, 48, 54 ), pool_size_1 = ( 2, 2 ), 
		filter_shape_2 = ( 50, 20, 5, 4 ), image_shape_2 = ( 10, 20, 22, 25 ), pool_size_2 = ( 2, 2 ),
		hiddenLayer_input_d = 50*9*11,
		hiddenlayer_output_d = 500,
		n_class = 2 )

setY=[]
for i in xrange(0, 51):
	setY.append(1)


instanceEx01.setData( ( 48, 54 ), 51, 51, 51, "test.mrc", "test.mrc", "test.mrc", [10,61], [10,61], [10,61], setY, setY, setY )

instanceEx01.trainingStart( learning_rate = 0.1, epochN = 500, batch_size = 10 )

instanceEx01.writeTrainingLocalMinimum( "convRecord_Ex02.dat" )
print( "... So I take it that the Ex02 finally works." )
