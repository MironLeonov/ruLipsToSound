import os
import os.path as P

video_name = 'posner'
glob_duration = 62

video_path = f'data/videos_23fps/{video_name}.mp4'
audio_path = f'data/audio_16000hz/posner{video_name}.wav'
output_dir = 'data/features_data'



if __name__ == '__main__': 
    start = 0
    idx = 0
    os.makedirs(output_dir, exist_ok=True)
    
    audio_features_dir = P.join(output_dir, "audio")
    video_features_dir = P.join(output_dir, "video")

    os.makedirs(audio_features_dir, exist_ok=True)
    os.makedirs(video_features_dir, exist_ok=True)


    for i in range(glob_duration): 
        video_name = f'{i}.mp4'
        audio_name = f'{i}.wav'
        os.system(f'ffmpeg -ss {i} -i {video_path} -t {1} -c copy {P.join(video_features_dir, video_name)}')
        os.system(f'ffmpeg -ss {i} -i {audio_path} -t {1} -c copy {P.join(audio_features_dir, audio_name)}')

