import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
cumError=0 
lastError=0
previousTime=0
def computePID(inp,cumError,lastError):
	global previousTime
	kp = 3
	ki = 5
	kd = 1
	#def nothing(x):
	 #   pass

	# Create a black image, a window
	#img2 = np.zeros((300,512,3), np.uint8)
	#cv2.namedWindow('pid tuner')

	# create trackbars for color change
	#cv2.createTrackbar('p','image2',0,1000,nothing)
	#cv2.createTrackbar('i','image2',0,1000,nothing)
	#cv2.createTrackbar('d','image2',0,1000,nothing)

	# create switch for ON/OFF functionality
	#switch = '0 : OFF \n1 : ON'
	#cv2.createTrackbar(switch, 'image2',0,1,nothing)
	#cv2.imshow('image2',img2)
	    # get current positions of four trackbars
	#kp = cv2.getTrackbarPos('p','image2')
	#ki = cv2.getTrackbarPos('i','image2')
	#kd = cv2.getTrackbarPos('d','image2')
	#s = cv2.getTrackbarPos(switch,'image2')
	setpoint=0
	currentTime = time.time()   
	                #//get current time\
	print("currentTime",currentTime)
	print("previous",previousTime)
	elapsedTime = (currentTime - previousTime)        #//compute time elapsed from previous computation
	print('elapsedTime',elapsedTime)
	error = int(setpoint-inp)                               # // determine error
	cumError += error * elapsedTime                #// compute integral
	rateError = (error-lastError)/elapsedTime   #// compute derivative

	out = (kp*error + ki*cumError + kd*rateError)/10               # //PID outputa               

	lastError = error
	print("lastError",lastError)                               # //remember current error
	previousTime = currentTime                     # //remember current time

	return math.floor(out)
def make_coordinates(image, line_parameters):
	try:
		slope,intercept = line_parameters
		y1 = image.shape[0]
		y2=int(y1*(3/6))
		x1=int((y1-intercept)/slope)
		x2=int((y2-intercept)/slope)
		return np.array([x1,y1,x2,y2])
	except:
		return np.array([0,0,0,0])

def average_slope_intercept(image, lines):
	left_fit=[]
	right_fit=[]
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2= line.reshape(4)
			parameters = np.polyfit((x1,x2),(y1,y2),1)
			slope = parameters[0]
			intercept= parameters[1]
			if slope<=0:
				left_fit.append((slope, intercept))
			else:
				right_fit.append((slope, intercept))

	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)
	left_line= make_coordinates(image, left_fit_average)
	right_line=make_coordinates(image, right_fit_average)
	#print(left_line,right_line)
	return np.array([left_line, right_line]), left_fit_average, right_fit_average
def canny(image):
	gray= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur=cv2.GaussianBlur(gray, (5,5),0)
	canny=cv2.Canny(blur,50,150)
	return canny

def  display_lines(image, lines):
	try:
		line_image = np.zeros_like(image)
		if lines is not None:
			for x1,y1,x2,y2 in lines:
				cv2.line(line_image, (x1,y1),(x2,y2),[255,255,255],1)
		return line_image
	except:
		pass

def region_of_interest(image):
	height=image.shape[0]
	polygons = np.array([[(0, 450),(image.shape[1], 450),(490,360),(780,360)]])
	mask= np.zeros_like(image)
	cv2.fillPoly(mask, polygons, 255)
	masked_image = cv2.bitwise_and(image,mask)
	cv2.imshow('a',masked_image)
	cv2.waitKey(0)
	return masked_image

cap= cv2.VideoCapture("/home/raj/Desktop/sovereign_vehicle/advv3.mp4")
#def lane_det(lane_image):

while(cap.isOpened()):
	ret,lane_image=cap.read()
	#print(np.shape(lane_image))
	canny_image=canny(lane_image)
	#cropped_image = region_of_interest(canny_image)
	lines = cv2.HoughLinesP(canny_image,2,np.pi/180,100, minLineLength=50, maxLineGap=100)
	#if lines is not None:
	#		for x1,y1,x2,y2 in lines[0]:
	#			cv2.line(lane_image, (x1,y1),(x2,y2),[255,0,0],10)
	#print(lines)
	try:
		averaged_lines, (left_slpoe, left_intercept), (right_slope, right_intercept) = average_slope_intercept(lane_image, lines)
	except:
		pass
	# print(left_intercept, right_intercept)
	#print(averaged_lines)
	line_image = display_lines(lane_image, averaged_lines)

	#print(np.shape(line_image))
	#combo_image = cv2.addWeighted(lane_image, 0.8 , line_image, 1 ,1)
	try: 
		combo_image = cv2.addWeighted(lane_image, 0.8 , line_image, 1 ,1)
		# cv2.imshow("weighted_add", combo_image)
		cv2.line(line_image, (256,512),(256,0),[255,255,255],1)

		# cv2.imshow("result", line_image)
		x_cen=256
		right_line_y = (right_slope*512) + right_intercept
		#print(left_intercept, right_line_y)
		z=min(left_intercept, right_line_y)
		#y_cen=min(left_intercept, right_line_y)
		y_cen=math.floor(z)
		print(y_cen)
		i=0
		j=511
		while (i<=x_cen):
			l=line_image[y_cen][i]
			if(l[0]==255 and l[1]==255 and l[2]==255):
				#print ('left',i)
				break
			else:
				i=i+1
		while (j>=x_cen):
			r=line_image[y_cen][j]
			if(r[0]==255 and r[1]==255 and r[2]==255):
				#print('right',j)
				break
			else:
				j=j-1
		offset=256-((i+j)/2)
		aa=((i+j)//2)

		print (offset)
		cv2.circle(line_image,(aa,y_cen), 5, (0,0,255), -1)
		cv2.circle(line_image,(256,y_cen),5,(255,0,0),-1)
		cv2.line(line_image, (aa,y_cen),(256,y_cen),[0,255,0],5)
		cv2.imshow('output',line_image)
		#return offset
	except:
		pass
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()