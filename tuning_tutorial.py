from cv2 import *
import numpy as np
import sys
from pynetworktables import *

# wpilib crashes if you don't do this.. 
SmartDashboard.init()

class ImageProcessor:
  hue_thresh = 100
  sat_thresh = 221
  val_thresh = 135
  max_thresh = 255
  hsv_calced = False

  def __init__(self, img_path):
    self.img_path = img_path

  def process(self):
    self.img       = imread(self.img_path)
    self.source_title   = self.img_path         
    self.h_title        = "hue"                 
    self.s_title        = "sat"                 
    self.v_title        = "val"                 
    self.combined_title = "combined_thresholds" 

    hsv = cvtColor(self.img, cv.CV_BGR2HSV)
    self.h, self.s, self.v = split(hsv)
    self.update_hue_threshold(self.hue_thresh)
    self.update_sat_threshold(self.sat_thresh)
    self.update_val_threshold(self.val_thresh)
    self.hsv_calced = True
    self.update_combined()
    self.layout_result_windows(self.h,self.s,self.v)
   
    waitKey(0)

  def layout_result_windows(self, h, s, v):
    pos_x, pos_y        = 500,500               
    imshow(self.img_path, self.img)
    imshow(self.h_title, h)
    imshow(self.s_title, s)
    imshow(self.v_title, v)
    imshow(self.combined_title, self.combined)

    moveWindow(self.h_title, pos_x*1, pos_y*0);
    moveWindow(self.s_title, pos_x*0, pos_y*1);
    moveWindow(self.v_title, pos_x*1, pos_y*1);
    moveWindow(self.combined_title, pos_x*2, pos_y*0);

    createTrackbar( "Hue Threshold:", self.source_title, self.hue_thresh, self.max_thresh, self.update_hue_threshold);
    createTrackbar( "Sat Threshold:", self.source_title, self.sat_thresh, self.max_thresh, self.update_sat_threshold);
    createTrackbar( "Val Threshold:", self.source_title, self.val_thresh, self.max_thresh, self.update_val_threshold);

  def update_hue_threshold(self, thresh):
    delta = 15
    self.h_clipped = self.threshold_in_range(self.h, thresh-delta, thresh+delta)
    imshow(self.h_title, self.h_clipped)
    self.update_combined()
    SmartDashboard.PutString("hue threshold:",str(thresh))

  def update_sat_threshold(self, thresh):
    delta = 67
    self.s_clipped = self.threshold_in_range(self.s, thresh-delta, thresh+delta)
    imshow(self.s_title, self.s_clipped)
    self.update_combined()

  def update_val_threshold(self, thresh):
    delta = 239
    self.v_clipped = self.threshold_in_range(self.v, thresh-delta, thresh+delta)
    imshow(self.v_title, self.v_clipped)
    self.update_combined()


  def threshold_in_range(self, img, low, high):
    unused, above = threshold(img, low, self.max_thresh, THRESH_BINARY)
    unused, below = threshold(img, high, self.max_thresh, THRESH_BINARY_INV)
    return bitwise_and(above, below)

  def update_combined(self):
    #combine all the masks together to get their overlapping regions.
    if (self.hsv_calced): 
      self.combined = bitwise_and(self.h_clipped, bitwise_and(self.s_clipped, self.v_clipped))
      imshow(self.combined_title, self.combined)

if '__main__'==__name__:
  try:
    img_path = sys.argv[1]
  except:
    print('Please add an image path argument and try again.')
    sys.exit(2)

  ImageProcessor(img_path).process()


