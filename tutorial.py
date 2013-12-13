import cv2
import numpy as np
import sys

class ImageProcessor:
  hue_thresh = 100
  max_thresh = 255

  def __init__(self, img_path):
    self.img_path = img_path

  def process(self):
    img = cv2.imread(self.img_path)
    cv2.imshow(self.img_path, img)

    hsv = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
    h, s, v = cv2.split(hsv)
    self.layout_result_windows(h,s,v)
   
    cv2.waitKey(0)

  def layout_result_windows(self, h,s,v):
    pos_x, pos_y = 500,500
    cv2.imshow("hue inverted", h)
    cv2.moveWindow('h', pos_x*1, pos_y*0);
    cv2.imshow("sat", s)
    cv2.moveWindow('s', self.pos_x*0, self.pos_y*1);
    cv2.imshow("val", v)
    cv2.moveWindow('v', self.pos_x*1, self.pos_y*1);
    cv2.createTrackbar( "Hue Threshold:", self.img_path, self.hue_thresh, self.max_thresh, self.hue_thresh_callback );
 

  def hue_thresh_callback(self, thresh):
    print('in hue_thresh_callback')
    # cv2.threshold(hsv_channels[0], red_hue_threshold_output, hue_thresh-15, 255, THRESH_BINARY ); 
    # cv2.threshold(hsv_channels[0], hue_threshold_output, hue_thresh+15, 255, THRESH_BINARY_INV );


if '__main__'==__name__:
  try:
    img_path = sys.argv[1]
  except:
    print('Please add an image path argument and try again.')
    sys.exit(2)

  ImageProcessor(img_path).process()


