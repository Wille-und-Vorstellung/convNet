# -*- coding: utf-8 -*-
'''
ID: dataLoader Ex00
Desciption: 
	loading mrc files( or mrcs file ) into compatible format for convNet as input and during which executing image processing algorithms
Input:
	
Output:

Date:

Notice:

'''

__author__ = "Ruogu Gao"

from EMAN2 import *
import os
import sys
import logging
import numpy
import theano
import theano.tensor as T
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logging.basicConfig( filename = 'dataLoader.log', level = logging.DEBUG )

'''
Phase II schedule:


'''

class dataLoader(object):
	'''
	prototype mrc loader for convNet
	'''

	def __init__( self, imageSize, imageN_test, imageN_train, imageN_validate ):
		#initialize component
		'''
		:type imageSize: tuple, (width, height)
		:param imageSize: size of test,train,validate image

		:type imageN_*: int
		:param imagN_*: image number in corresponding set

		'''
		self._test_data_x = numpy.ndarray( shape=( imageN_test, imageSize[0]*imageSize[1]), dtype='float64', order='C' )
		self._test_data_y = numpy.ndarray( shape=imageN_test, dtype='int64' )
		self._train_data_x = numpy.ndarray( shape=( imageN_train, imageSize[0]*imageSize[1]), dtype='float64', order='C' )
		self._train_data_y = numpy.ndarray( shape=imageN_train, dtype='int64' )
		self._validate_data_x = numpy.ndarray( shape=( imageN_validate, imageSize[0]*imageSize[1]), dtype='float64', order='C' )
		self._validate_data_y = numpy.ndarray( shape=imageN_validate, dtype='int64' )

		self._imageSize = imageSize
		self._imageN_test = imageN_test
		self._imageN_train = imageN_train
		self._imageN_validate = imageN_validate

		self._test_mrc = EMData()
		self._train_mrc = EMData()
		self._validate_mrc = EMData()

		self._data_ready = False
		self._preproccessed = False
		
	def setData( self, testPath, trainPath, validatePath ):
		##all three input should be path of mrc file that contains multiple images( formally should be mrcs )
		##and we assume these input are of correct format
		##Phase II: verify those pathes

		self._test_mrc.read_image( testPath )
		self._train_mrc.read_image( trainPath )
		self._validate_mrc.read_image( validatePath )
		self._data_ready = True

		print("... MRC images reading complete")
		logging.info( "... MRC images reading complete" )
		

	def loadData(self, testRange, trainRange, validateRange,
		testY, trainY, validateY):
		##return converted data sets in a tuple, which should include test, train, validate sets
		##the interval given are taken as [ , )
		'''
		:type *Y: list
		:param *Y: the class label of corresponding data

		:type *range: tuple
		:param *rannge: indicates the selected images to transform in the mrc files 

		'''
		#check input range
		if ( testRange[0] < 0 or trainRange[0] < 0 or validateRange[0] < 0 ):
			_errorMessage()
			logging.warning( "invalid input range at loadData()" )
		elif ( testRange[1] > self._test_mrc.get_zsize() or 
			trainRange[1] > self._train_mrc.get_zsize() or 
			validateRange[1] > self._validate_mrc.get_zsize() ):
			_errorMessage()
			logging.warning( "invalid input range at loadData()" )
		else:
			pass
		#reorganise
		for k in xrange( testRange[0], testRange[1] ):
			for i in xrange( 0, self._test_mrc.get_xsize() ):
				for j in xrange( 0, self._test_mrc.get_ysize() ):
					self._test_data_x[k - testRange[0]][i*self._imageSize[0] + j] = self._test_mrc.get_value_at(i, j, k) + 10

		self._test_data_y = numpy.asarray( testY, dtype='int64' )

		for k in xrange( trainRange[0], trainRange[1] ):
			for i in xrange( 0, self._train_mrc.get_xsize() ):
				for j in xrange( 0, self._train_mrc.get_ysize() ):
					self._train_data_x[k - testRange[0] ][i*self._imageSize[0] + j] = self._train_mrc.get_value_at(i, j, k) + 10
		
		self._train_data_y = numpy.asarray( trainY, dtype='int64' )

		for k in xrange( validateRange[0], validateRange[1] ):
			for i in xrange( 0, self._validate_mrc.get_xsize() ):
				for j in xrange( 0, self._validate_mrc.get_ysize() ):
					self._validate_data_x[k - testRange[0] ][i*self._imageSize[0] + j] = self._validate_mrc.get_value_at(i, j, k) + 10

		self._validate_data_y = numpy.asarray( validateY, dtype='int64' )

		#set to shared variables for potential GPU usage
		#shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
		#T.cast(shared_y, 'int32')
		self._shared_test_data_x  = theano.shared(numpy.asarray(self._test_data_x, dtype=theano.config.floatX), borrow=True)
		self._shared_test_data_y_tmp = theano.shared(numpy.asarray(self._test_data_y, dtype=theano.config.floatX), borrow=True)
		self._shared_test_data_y = T.cast( self._shared_test_data_y_tmp, 'int64' )

		self._shared_train_data_x = theano.shared(numpy.asarray(self._train_data_x, dtype=theano.config.floatX), borrow=True)
		self._shared_train_data_y_tmp = theano.shared(numpy.asarray(self._train_data_y, dtype=theano.config.floatX), borrow=True)
		self._shared_train_data_y = T.cast( self._shared_train_data_y_tmp, 'int64' )

		self._shared_validate_data_x = theano.shared(numpy.asarray(self._validate_data_x, dtype=theano.config.floatX), borrow=True)
		self._shared_validate_data_y_tmp = theano.shared(numpy.asarray(self._validate_data_y, dtype=theano.config.floatX), borrow=True)
		self._shared_validate_data_y = T.cast( self._shared_validate_data_y_tmp, 'int64' )
		
		print("... Data reorganizing complete")
		logging.info("... Data reorganizing complete")
		result =[( self._shared_test_data_x, self._shared_test_data_y ), 
					( self._shared_train_data_x, self._shared_train_data_y ), 
					( self._shared_validate_data_x, self._shared_validate_data_y )]
		return result

	def preprocessing(self):
		##customised preprocessing option
		# WIP

		self._preproccessed = True

		
	def _errorMessage(self):
		print( "...Some error occured, please refer to the log file for detail info" )


####------------------------Nirn sub-plane
#testing sequence using pylab
#setY=[]
#for i in xrange(0, 76):
	#setY.append(1)

#instance1 = dataLoader( (48, 54), 76, 76, 76 ) 
#instance1.setData( "test.mrc", "test.mrc", "test.mrc" )
#dataset = instance1.loadData( [60,71], [60,71], [60,71], setY, setY, setY )


#img = dataset[0][0]
#print( img )

#plot = plt.imshow( img, norm=matplotlib.colors.Normalize() )

