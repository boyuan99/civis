import pandas as pd
import numpy as np
import subprocess
import math
from scipy.signal import find_peaks, savgol_filter, butter, sosfilt, sosfiltfilt, medfilt, peak_widths
from bokeh.plotting import figure, output_file, save, reset_output, show
from bokeh.io import output_notebook,show
from bokeh.models import Span, VStrip
from bokeh.layouts import column,row
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# [0, 0] is top left

class ProssessedPostureAnalysis():
    def __init__(self, loc_paths, video_paths):
        self.loc_paths = loc_paths
        self.data_sheets = []
        self.left_or_right = []
        self.plotting_arr = []
        self.optimal_thresholds = 0.0
        self.optimal_true_positives = 0.0
        self.optimal_false_positives = 0.0
        self.optimal_true_negatives = 0.0
        self.optimal_false_positives = 0.0
        self.videos = []
        for i in range(len(video_paths)):
            self.videos.append(VideoFileClip(video_paths[i]))


        # add all data into the object in the format of DataFrames
        for path in loc_paths:
            df = pd.read_csv(path)
            self.data_sheets.append(df)
        
        # classify left and right
        for i in range(len(loc_paths)):
            curr_lr = []
            curr_data_sheet = self.data_sheets[i]
            for y in range(2,curr_data_sheet.shape[0]):
                low_diag_line_m = -(270 - float(curr_data_sheet.iloc[y,2]))/float(curr_data_sheet.iloc[y,1])
                low_diag_line_b = 270
                high_diag_line_m = -(180 - float(curr_data_sheet.iloc[y,2]))/float(curr_data_sheet.iloc[y,1])
                high_diag_line_b = 175
                y_threshold = [low_diag_line_m * float(curr_data_sheet.iloc[y,10]) + low_diag_line_b,high_diag_line_m * float(curr_data_sheet.iloc[y,10]) + high_diag_line_b]
                
                value = float(curr_data_sheet.iloc[y, 11])  # Convert the value to float

                if (y_threshold[0] > value > y_threshold[1]) and float(curr_data_sheet.iloc[y,10]) < 122:
                    curr_lr.append("center")
                elif value < y_threshold[0]:
                    curr_lr.append("left")
                else:
                    curr_lr.append("right")
                    self.plotting_arr.append(0)
            self.left_or_right.append(curr_lr)

        for j in range(len(self.videos)):
            video_w, video_h = self.videos[j].size
            
            # Create an empty array to hold all frames
            num_frames = int(self.videos[j].fps * self.videos[j].duration)
            frames_array = np.empty((num_frames, video_h, video_w, 3), dtype=np.uint8)
            
            # Process and store each frame in the array
            for i, frame in enumerate(self.videos[i].iter_frames(fps=40, dtype='uint8')):
                t = i / self.videos[j].fps
                processed_frame = self.add_text_to_frame(frame, j, t, video_w, video_h)
                frames_array[i] = processed_frame
            
            output_video_path = "left_or_right_" + str(j) + ".avi"
            # Save the frames array as an AVI video using ffmpeg
            self.save_frames_as_video(frames_array, output_video_path, video_w, video_h, 40)

    def add_text_to_frame(self,frame, i, t, video_w, video_h):
        txt = self.left_or_right[i][int(t * 40)]
        text_clip = TextClip(txt, fontsize=12, color='white', bg_color='black', size=(video_w, None)).set_position(('center', 'bottom'))
        text_frame = text_clip.get_frame(t)
        text_frame = np.array(text_frame)
        
        # Ensure text frame has the same number of channels as the video frame
        if text_frame.shape[2] == 4:  # If text frame has alpha channel
            text_frame = text_frame[:, :, :3]  # Remove alpha channel
        frame = np.array(frame, copy=True)
        
        text_h, text_w, _ = text_frame.shape
        y_offset = video_h - text_h  # Position the text at the bottom
        x_offset = (video_w - text_w) // 2  # Center the text horizontally
        
        # Blend the text frame on top of the original frame
        for y in range(text_h):
            for x in range(text_w):
                if text_frame[y, x].sum() > 0:  # If the text pixel is not black
                    frame[y + y_offset, x + x_offset] = text_frame[y, x]
        
        return frame
    
    def save_frames_as_video(self, frames_array, output_video_path, video_w, video_h, fps):
        # Prepare ffmpeg command to write video
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',  # Input format
            '-vcodec', 'rawvideo',  # Input codec
            '-s', f'{video_w}x{video_h}',  # Input resolution
            '-pix_fmt', 'rgb24',  # Input pixel format
            '-r', str(fps),  # Input frame rate
            '-i', '-',  # Input from stdin
            '-an',  # No audio
            '-c:v', 'mpeg4',  # Output codec
            output_video_path
        ]
        
        # Open ffmpeg process
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
        
        # Write frames to ffmpeg process
        process.stdin.write(frames_array.tobytes())
        
        # Close stdin and wait for process to complete
        process.stdin.close()
        process.wait()
                            



def main():
    paths = ["/Users/edward41803/Documents/CIS_Processed_Data/PostureEstimation/PostureEstimation-BY-2024-06-04/367_06062024-06062024135704-0000DLC_resnet50_PostureEstimationJun4shuffle1_30000.csv"]
    vid_paths = ["/Users/edward41803/Documents/CIS_Processed_Data/PostureEstimation/PostureEstimation-BY-2024-06-04/367_06062024-06062024135704-0000.avi"]
    process_posture = ProssessedPostureAnalysis(paths, vid_paths)
    
    

    
if __name__ == "__main__":
    main()
