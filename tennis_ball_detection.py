import cv2
import argparse
from process_video import process_video as pv
import numpy as np
import math

# HSV bounds for green
LOWER_GREEN = (29, 86, 6)
UPPER_GREEN = (64, 255, 255)
class tennis_ball_detection:

    def __init__(self, args):
        self.video_name = args['input_video']
        self.slopeA = 0
        self.slopeB = 0
        self.interceptA = 0
        self.interceptB = 0
        self.slowmo = args['s']
        
        
    def detect_tennis_ball(self):
        """
        Purpose: Main Function. Calculates the line of equations for the region boundaries.
        Detects tennis ball and track location in the four regions.
        
        
        Args:
            None
        Output:
            .avi video of the rolling tennis ball with a bounding circle laid on top of the
            tennis ball and which region the ball is in.
        """
        process_frame = pv.read_video(self.video_name)
        current_frame = next(process_frame)
        self.detect_regions(current_frame)
            
        print("The detected equation of the positively sloped line is: y=", self.slopeA, "x + ", self.interceptA)
        print("The detected equation of the negatively sloped line is: y=", self.slopeB, "x + ", self.interceptB)
        (h,w, _) = current_frame.shape
        video_out = pv.write_video("Zonera's_TennisBall_Detection_Demo.avi", (w,h))
        
        i = 1
        while (current_frame is not None):
            clone = current_frame.copy() # this copy of the original frame is used for the final image that will be written out
            current_frame = cv2.GaussianBlur(current_frame, (3,3), 0)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(current_frame, LOWER_GREEN, UPPER_GREEN) #create mask based on the green bounds
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                
                c = max(contours, key=cv2.contourArea)
                ((x,y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 50 and radius < 350: # assumed size of the radius
                    cv2.circle(clone, (int(center[0]), int(center[1])), int(radius), (0, 0,255), 2)
                    cv2.circle(clone, (int(center[0]), int(center[1])), 2, (0, 0,0), 3)
                    deteremined_location = self.determine_region(center)
                    if deteremined_location == 0:
                        print ("We were unable to locate the tennis ball.")
                        cv2.putText(clone, "We were unable to locate the ball.", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                    else:
                        print("The ball is currently located in region ", deteremined_location)
                        cv2.putText(clone, "Current Region: " + str(deteremined_location), (100,100), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2)
                    if self.slowmo:
                        cv2.imshow('lines', clone)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    print("We were not able to find the tennis ball.") #the detected contours didn't "look" like a tennis ball based on assumptions on the size of the ball
            video_out.write(clone)
            i = i + 1
            current_frame = next(process_frame)
        video_out.release()
        
    def detect_regions(self, current_frame):
        """
        Purpose: Calculates the line of equations for the region boundaries. Lines are
        detected using a probabilistic hough transform.
        
        In the real world, you could recalculate these boundaries at some interval. For this
        assignment, I call this function at the start and assume the changes in the camera angle
        are minimal.
        
        Args:
            current_frame - numpy array
        Output:
            None
        """
        
        if self.slopeA == 0 or self.slopeB == 0:
            working_frame = cv2.GaussianBlur(current_frame, (3,3), 0)
            working_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
            working_frame[working_frame < 200] = 0 #since the lines are white, we can use a high global threshold to create a mask
            edges = cv2.Canny(working_frame,50,150, apertureSize = 3)
            lines = cv2.HoughLinesP(edges,1,np.pi/180, 15, np.array([]), 50, 120)
            # since the lines start at the corners of the frame, we can use the min and max to create our bounding lines for the regions;
            minX = min(lines[:,:,0])[0]
            minY = min(lines[:,:,1])[0]
            maxX = max(lines[:,:,2])[0]
            maxY = max(lines[:,:,3])[0]
            
            cv2.line(current_frame,(minX,minY),(maxX,maxY),(0,0,255),2) # negative line
            cv2.line(current_frame,(minX,maxY),(maxX,minY),(0,0,255),2) # positive line
            
            # using the standard equation of a line (delta Y / delta X)
            slopeA = ((maxY - minY) / (maxX - minX)) # negative slope
            slopeB = ((minY - maxY) / (maxX - minX)) # positive slope
            interceptA = minY - minX*slopeA
            interceptB = maxY - minX*slopeB
            
            self.slopeA = slopeA
            self.slopeB = slopeB
            self.interceptA = interceptA
            self.interceptB = interceptB
            
        
    def determine_region(self, tennis_ball_centroid):
        """
        Purpose: Given the (x,y) coordinates of the tennis ball, this function determines
        which region the tennis ball is currently in.
        
        Since we know the equations for both lines in the frame, we can calculate y by using the x
        value for the tennis ball into our equations and compare the resulting y values against
        the y coordinate of the tennis ball to determine which region the tennis ball is in.
        
        Note: In python, the origin is in the top left corner. Therefore, our inequalities are
        less intuitive.
        
        Args:
            tennis_ball_centroid - (x, y) coordinate
        Output:
            region value - int
        """
        tbx, tby = tennis_ball_centroid
        
        calc_1_y = self.slopeA*tbx + self.interceptA # negative slope
        calc_2_y = self.slopeB*tbx + self.interceptB # positive slope
        
        if tby <= calc_2_y and tby > calc_1_y:
            return 1
        elif tby <= calc_2_y and tby <= calc_1_y:
            return 2
        elif tby > calc_2_y and tby <= calc_1_y:
            return 3
        elif tby > calc_2_y and tby > calc_1_y:
            return 4
        else:
            return 0 # an error occurred
            
        
    
        
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-input_video", "-input_video", required=True, help="Input video for detecting tennis ball")
    ap.add_argument("-s", action="store_true", help="Including this flag allows you to click through video frame by frame")
    args = vars(ap.parse_args())
    if args["input_video"] == "":
        print ("User did not provide an input_video.".format(args["speed"]))
        sys.exit(0)
    cd = tennis_ball_detection(args)
    cd.detect_tennis_ball()
