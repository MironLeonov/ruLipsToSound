import os
import os.path as P


video_path = 'posner.mp4'
output_dir = 'data'
fps = 23
cut_video_dir = P.join(output_dir, 'videos_algin')
sr = '16000'

if __name__ == '__main__': 
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)
    audio_name = video_name.replace(".mp4", ".wav")

    ori_audio_dir = P.join(output_dir, "audio_ori")
    os.makedirs(ori_audio_dir, exist_ok=True)
    os.system(f"ffmpeg -i {video_path} -loglevel error -f wav -vn -y {P.join(ori_audio_dir, audio_name)}")

    sr_audio_dir = P.join(output_dir, f"audio_{sr}hz")
    os.makedirs(sr_audio_dir, exist_ok=True)
    os.system("ffmpeg -i {} -loglevel error -ac 1 -ab 16k -ar {} -y {}".format(
        P.join(ori_audio_dir, audio_name), sr, P.join(sr_audio_dir, audio_name)))


    # change video fps
    fps_audio_dir = P.join(output_dir, f"videos_{fps}fps")
    os.makedirs(fps_audio_dir, exist_ok=True)
    os.system(f"ffmpeg -y -i {video_path} -loglevel error -r {fps} -c:v libx264 -strict -2 {P.join(fps_audio_dir, video_name)}")