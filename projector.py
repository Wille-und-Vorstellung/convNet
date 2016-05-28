#make projections, just that.

from EMAN2 import *
import sys
import os
import numpy
import logging
import cPickle as pickle

logging.basicConfig( filename = 'projector.log', level = logging.DEBUG )
'''
def mergeEM( origin, append):
	pass
'''
class mergable(object):
	"""docstring for mergable"""

	def __init__(self, size):
		self._data = EMData()
		self._zCounter = 0
		self._data.set_size( nx = size[0], ny = size[1], nz = size[2] )
		self._data.to_zero()
		self._size = [0,0,0]
		self._size[0] = size[0]
		self._size[1] = size[1]
		self._size[2] = size[2]

	def mergeEM2d(self, append):

		for x in xrange(0, append.get_xsize()):
			for y in xrange(0, append.get_ysize()):
				tmp = append.get_value_at( x, y, 0 )
				self._data.set_value_at( x, y, self._zCounter, tmp )

		self._zCounter += 1
		logging.info( "... mergeEM ready, Z="+str( self._zCounter ) )

		
	def get(self):
		return self._data

	def set(self, newComer):
		self._data = newComer

	def getSize(self):
		return self._size

	def mergeEM3d(self, append):
		#append to the rail and require the 2d image be same size
		#return a new one and change the old one as well
		#check 
		if ( append.get_xsize() != self._data.get_xsize() ) or ( append.get_ysize() != self._data.get_ysize() ):
			print("error encountered, shutting down...")
			logging.info( "invalid input 3d image size" )
			exit()
		#get down to bussiness
		self._size[2] += append.get_zsize()
		merged3d = EMData()
		merged3d.set_size( self._data.get_xsize(), self._data.get_ysize(), self._size[2])
		merged3d.to_zero()

		for z in xrange(0, self._size[2]):
			for x in xrange(0, self._data.get_xsize()):
				for y in xrange(0, self._data.get_ysize()):
					if ( z < self._data.get_zsize() ):
						tmp = self._data.get_value_at(x,y,z)
					else:
						tmp = append.get_value_at(x, y, ( z - self._data.get_zsize() ) )
					merged3d.set_value_at(x, y, z, tmp)

		self._data = merged3d

		logging.info("... mergeEM3d complete")
		print("... mergeEM3d complete")
		return merged3d
		
	def slice(self, sliceSize, step=1):
		#slice into new EMData which shall replace the old one
		#check
		if ( sliceSize[0] >= self._data.get_xsize() ) \
			or ( sliceSize[1] >= self._data.get_ysize() ):
			print("error encountered, shutting down...")
			logging.info( "invalid input slice size" )
			exit()

		multipler = [0,0]
		multipler[0] = (self._data.get_xsize() - sliceSize[0])/step + 1
		multipler[1] = (self._data.get_ysize() - sliceSize[1])/step + 1

		sliced = EMData()
		sliced.set_size(sliceSize[0], sliceSize[1], self._data.get_zsize()*multipler[0]*multipler[1] )
		sliced.to_zero()
		
		tmpz=0
		tmp=0
		for z in xrange(0, self._data.get_zsize()):
			for i in xrange(0, multipler[0]):
				for j in xrange(0, multipler[1]):
					for x in xrange(i*step, sliceSize[0]+i*step):
						for y in xrange(j*step, sliceSize[1]+j*step):
							tmp = self._data.get_value_at(x, y, z)
							sliced.set_value_at(x-i*step, y-j*step, tmpz, tmp ) ####
					tmpz+=1

		self._data = sliced
		logging.info("... slice done")
		return sliced
		

	def reshape(self, newSize):#input 2-tuple
		#check 
		if ( newSize[0] <= self._size[0] ) or ( newSize[1] <= self._size[1] ):
			print("error encountered, shutting down...")
			logging.info( "invalid reshaping size" )
			exit()

		#
		newshaped = EMData()
		newshaped.set_size(newSize[0], newSize[1], self._data.get_zsize())
		newshaped.to_zero()
		#stuff empty entry to the right and buttom of origin image
		for z in xrange(0, self._size[2]):
			for x in xrange(0, newSize[0]):
				for y in xrange(0, newSize[1]):
					if (x < self._size[0] ) and ( y < self._size[1] ): #copy old data
						tmp = self._data.get_value_at(x, y, z)
					else:
						tmp = 0
					newshaped.set_value_at(x, y, z, tmp)

		self._data = newshaped
		self._size[0] = newSize[0]
		self._size[1] = newSize[1]
		logging.info( "... reshape complete" )

	def subset(self, zlist): ####################
		#check
		if ( len(zlist) == 0  ) or ( len(zlist) > self._data.get_zsize() ):
			print("error encountered, shutting down...")
			logging.info( "invalid subset selection" )
			exit()
		#
		subset = EMData()
		subset.set_size( self._data.get_xsize(), self._data.get_ysize(), len(zlist) )
		tmpz=0
		for z in zlist:
			for x in xrange(0, self._data.get_xsize()):
				for y in xrange(0, self._data.get_ysize()):
					tmp = self._data.get_value_at(x, y, z)
					subset.set_value_at( x, y, tmpz, tmp )
			tmpz+=1

		self._data = subset
		self._size[2] = len(zlist)

		return subset


def projection( path, project_step, result_path ):
# return one single MRC that contain all the projections
# project_size: 2-tuple
	#reading MRC (transformed from PDB )
	model3d = EMData()
	model3d.read_image( path )
	
	#prepare the output projection container
	projection_n = (360/project_step)**3
	#check step
	
	if ( project_step <= 0 ) or ( project_step >=360 ):
		print("error encountered, shutting down...")
		logging.info( "invalid projection step" )
		exit()
	#projection start
	Gwyen = EMData()
	flag=False
	for x in xrange(0, 360, project_step):
		for y in xrange(0, 360, project_step):
			for z in xrange(0, 360, project_step):
				Gwyen = model3d.project( "standard", Transform( {"type":"eman", "alt":z, "az":y, "phi":x  }) )
				if ( flag == False ):
					projections = mergable(( Gwyen.get_xsize(), Gwyen.get_ysize(), projection_n))
					flag = True
					pSize = projections.getSize()
				if ( Gwyen.get_xsize() != pSize[0] ) or ( Gwyen.get_ysize() != pSize[1] ):
					print("Holy shit!!!"+str(x)+"|"+str(y)+"|"+str(z))
					logging.info("Holy shit!!!"+str(x)+"|"+str(y)+"|"+str(z))
					exit()
				projections.mergeEM2d( Gwyen )

	logging.info("... projection ready")
	result = projections.get()
	result.write_image( result_path )
	logging.info("... written to file")
	return result


def main(): #testing purpose
	'''
	projected = projection( "test3.mrc", 90, "test3proX0.mrc" )
	projectedx = projection( "test4.mrc", 90, "test4proX0.mrc" )
	print("... projection done")
	proShaped = mergable( (projected.get_xsize(), projected.get_ysize(), projected.get_zsize()) )
	proShaped.set( projected )
	proShaped.reshape( (218, 192) )
	tmpEM = proShaped.get()
	tmpEM.write_image( "test3proX0Shaped.mrc" )
	print("mergeEM3d test")
	blank = EMData()
	blank.set_size(projected.get_xsize(), projected.get_ysize(), 64)
	blank.to_zero()
	mergeTest = mergable( (projected.get_xsize(), projected.get_ysize(), projected.get_zsize()) )
	mergeTest.set( projected )
	mergeTest.mergeEM3d( blank )
	test3merged = mergeTest.get()
	test3merged.write_image("test3proX0Merged.mrc")
	
	print("slice test")
	sliceTest = mergable( (projectedx.get_xsize(), projectedx.get_ysize(), projectedx.get_zsize()) )
	sliceTest.set( projectedx )
	test4slice = sliceTest.slice( (92,152), step=10 )
	test4slice.write_image("test4proX0Sliced.mrc")
	'''
	print("... prepareting complex I data")

	train1 = projection( "fieldtest1part.mrc", 40, "train1.mrc" )
	valid1 = projection( "fieldtest1part.mrc", 90, "valid1.mrc" )
	test1 = projection( "fieldtest1whole.mrc", 90, "test1.mrc" )
	#slice
	tmpEM = mergable( (test1.get_xsize(), test1.get_ysize(), test1.get_zsize()) )
	tmpEM.set( test1 )
	test1Sliced = tmpEM.slice( ( train1.get_xsize(), train1.get_ysize()), step=40 )
	test1Sliced.write_image( "test1Sliced.mrc" )
	#megative data for train and validate set
	trainNeg1 = EMData()
	trainNeg1.set_size( train1.get_xsize(), train1.get_ysize(), 32 )
	trainNeg1.to_zero()
	trainNeg2 = EMData()
	trainNeg2.set_size( train1.get_xsize(), train1.get_ysize(), 32 )
	trainNeg2.to_value(1)


	validNeg1 = EMData()
	validNeg1.set_size( valid1.get_xsize(), valid1.get_ysize(), 32 )
	validNeg1.to_value(5)
	validNeg2 = EMData()
	validNeg2.set_size( valid1.get_xsize(), valid1.get_ysize(), 32 )
	validNeg2.to_value(12)
	# merge train set and validate set
	tmpEM = mergable( (train1.get_xsize(), train1.get_ysize(), train1.get_zsize()) )
	tmpEM.set( train1 )
	tmpEM.mergeEM3d( trainNeg1 )
	tmpEM.mergeEM3d( trainNeg2 )
	trainF1 =tmpEM.get()
	trainF1.write_image( "trainSet1.mrc" )
	tmpEM = mergable( (valid1.get_xsize(), valid1.get_ysize(), valid1.get_zsize()) )
	tmpEM.set( valid1 )
	tmpEM.mergeEM3d( validNeg1 )
	tmpEM.mergeEM3d( validNeg2 )
	validF1 =tmpEM.get()
	validF1.write_image( "validSet1.mrc" )
	#merge test set
	tmpEM = EMData()
	tmpEM.read_image( "test1Sliced.mrc" )
	testPos = mergable( (train1.get_xsize(), train1.get_ysize(), train1.get_zsize()) )
	testPos.set( tmpEM )
	
	#selected positive 30
	selectedList = [2,3,4,8,9,10,16,34,38,39,40,44,48,69,79,82,88,90,94,102,110,114,118,120,126,152,154,156,232,236]
	testPos.subset( selectedList )
	
	testNeg = EMData()
	testNeg.set_size( train1.get_xsize(), train1.get_ysize(), 30 ) 
	testNeg.to_value(1.5)

	testPos.mergeEM3d( testNeg )
	testF1 = testPos.get()
	testF1.write_image( "testSet1.mrc" )
	
	print( "... constructing label vectors" )
	trainY = []
	validY = []
	testY = []
	for i in xrange(0, 793):
		if (i < 729):
			tmp = 1
		else:
			tmp =0
		trainY.append(tmp)

	for j in xrange(0, 128):
		if (j < 64):
			tmp = 1
		else:
			tmp = 0 
		validY.append(tmp)

	
	for j in xrange(0, 60):
		if (j < 30) :
			tmp = 1
		else:
			tmp = 0 
		testY.append(tmp)
		
	print("... pickling")
	fr = open("trainY.label", "wb")
	ft = open("testY.label", "wb")
	fv = open("validY.label", "wb")

	pickle.dump(trainY, fr)
	pickle.dump(testY, ft)
	pickle.dump(validY, fv)
	
	fr.close()
	fv.close()
	ft.close()
	
	print("... let's find out how it gonna be")

	return 117

main()