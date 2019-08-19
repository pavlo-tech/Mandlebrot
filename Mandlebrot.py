from multiprocessing import Process
import matplotlib .pyplot as plt
from numba import jit
import threading
import numpy as np
import time


@jit
def ColorConverge(C: complex) -> []:
	Z = C
	for i in range(1,1000):
		Z = Z**2 + C
		if abs(Z) > 2:
			return [255, 255, 255]
	return [0, 0, 0]

@jit
def GenMandlebrot(rows: int, cols: int, stepSize: float) -> np.ndarray:
	arr = np.zeros((rows,cols,3))
	for r in range(rows):
		for c in range(cols):
			arr[r][c] =  ColorConverge(complex(stepSize*(c-cols/2),stepSize*(r-rows/2)))
	return arr


@jit
def GenMandlebrot_parallel_helper(arr: np.ndarray, rowStart:int, rowEnd:int, rows:int, cols: int, stepSize:float):
	sub_arr = np.zeros((rowEnd-rowStart,cols,3))
	for r in range(rowStart, rowEnd):
		for c in range(cols):
			arr[r][c] =  ColorConverge(complex(stepSize*(c-cols/2),stepSize*(r-rows/2)))

	arr

#@jit
def GenMandlebrot_parallel(rows: int, cols: int, stepSize: float, numcores: int = 4) -> np.ndarray:
	arr = np.zeros((rows,cols,3)); threads = []
	for i in range(numcores):
		threads.append(threading.Thread(target=GenMandlebrot_parallel_helper, 
										args=(arr, i*int(rows/numcores), (i+1)*int(rows/numcores), rows, cols, stepSize,)))
		threads[i].start()

	for i in range(numcores):
		threads[i].join()
	return arr

	
'''

start = time.time()
x=GenMandlebrot(10000,10000,.0003)
end = time.time()
print("non-parallel: "+str(end-start))
'''

'''
#basically no parallelism acheived
for i in range(1,256):
	start = time.time()
	x=GenMandlebrot_parallel(10000,10000,.0003,i)
	end = time.time()
	print("parallel ("+str(i)+" threads): "+str(end-start))
'''

'''
plt.imshow(x)
plt.show()
'''