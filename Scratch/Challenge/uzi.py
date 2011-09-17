from pylab import *
import datetime

MONTH = 1
DAY = 26

MONDAY = 0

for year in range(1006,2000,10):
#for year in range(1,2011):
    d = datetime.date(year,MONTH,DAY)
    if d.weekday() == MONDAY and d.year%4 == 0:
        print d

                 

image = imread('screen15.jpg')
