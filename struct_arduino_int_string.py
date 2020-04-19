import serial
import struct
import time
ser = serial.Serial('/dev/ttyACM0',9600)
print("straight")
s=6
while(1):

	if(s>5):

		ser.write(struct.pack('>B', s))
		s=s+1
	else:
		s=6
	if(s==20):
		s=1
	time.sleep(0.1)
	print(s)

