```
Video Frame Cutting is an essential task in video processing, often performed to extract crucial frames from video files.
The primary objective is to transform a video into a sequence of still images (frames) for research purposes.
The provided code snippet will assist you in cutting frames from videos.
```

import cv2
import os
import argparse

def export(input_dir, output_dir, frame_step):
    list_videos = os.listdir(input_dir)
    for video_number in list_videos:
        video_filename = os.path.join(input_dir, f'{video_number}')
        
        video_capture = cv2.VideoCapture(video_filename)
        
        if not video_capture.isOpened():
            print(f"Không thể mở video {video_number}")
            continue

        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_count % frame_step == 0:
                frame_filename = os.path.join(output_dir, f'{video_number}_{frame_count}.jpg')
                cv2.imwrite(frame_filename, frame)
                frame = cv2.resize(frame, (1280, 720))
                cv2.imshow('Frame', frame)
            
            frame_count += 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        video_capture.release()

    cv2.destroyAllWindows()

def get_output_folder():
    output_dir = 'frame' 
    os.makedirs(output_dir, exist_ok=True)
    default = 'output'
    list_name = os.listdir(output_dir)
    index = [int(item.replace('output', '')) for item in list_name]
    if index == []:
        output_dir = os.path.join(output_dir, default) + '1'
    else:
        output_dir = os.path.join(output_dir, default) + str(int(max(index)) + 1)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="cutting frames from video")
    parser.add_argument("--videos", required=True, help="path to the videos directory")
    parser.add_argument("--frame", required=True, help="frame step)")
    args = parser.parse_args()
    videos_path = args.videos
    frame_rate = args.frame
    frame_rate_dir = {
        'high': 1,
        'good': 2,
        'medium': 3,
        'low': 5
    }   
    if not os.path.exists(videos_path):
        print(f"Error: Videos directory '{videos_path}' does not exist.")
        return
    if frame_rate not in frame_rate_dir:
        print(f"Error: frame rate '{frame_rate}' does not exist.")
        print("Follow these:")
        print('\n'.join([f"--frame {key}" for key in frame_rate_dir.keys()]))
        return

    output_path = get_output_folder()
    export(videos_path, output_path, frame_rate_dir[frame_rate])

if __name__ == "__main__":
    main()
