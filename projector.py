#make projections, just that.

from EMAN2 import *
import sys
import os
import numpy
import logging

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
		self._size = size

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
		pass

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
		logging.info( "... reshape complete" )

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
	projected = projection( "test3.mrc", 90, "test3pro.mrc" )
	print("... projection done")
	proShaped = mergable( (projected.get_xsize(), projected.get_ysize(), projected.get_zsize()) )
	proShaped.set( projected )
	proShaped.reshape( (218, 192) )
	tmpEM = proShaped.get()
	tmpEM.write_image( "test3proShaped.mrc" )
	print("Injecting negative data")
	

	print("... let's find out how it gonna be")
	return 117

main()