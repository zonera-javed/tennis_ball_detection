import cv2

class process_video:
    
    def read_video(filename):
        video = cv2.VideoCapture(filename)
        
        while video.isOpened():
            ret, frame = video.read()
            
            if ret:
                yield frame
            else:
                break
        video.release()
        yield None
        cv2.destroyAllWindows()
        
    def write_video(filename, frame_size, fps=20):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        
        return cv2.VideoWriter(filename, fourcc, fps, frame_size)
    
        
