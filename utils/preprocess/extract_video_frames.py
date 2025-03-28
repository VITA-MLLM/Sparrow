import os
from tqdm import tqdm
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# for video-mme
# for category in Path('./').iterdir():
#     if not category.is_dir(): continue
#     for video_type in category.iterdir():
#         if not video_type.is_dir(): continue
#         video_dir = video_type.joinpath('video')
#         for video in video_dir.iterdir():
#             video_path = str(video)
#             save_path = os.path.join('extracted_frames', str(video_dir), video.stem)
#             os.makedirs(save_path, exist_ok=True)
            
#             extracted_command = f"ffmpeg -i {video_path} -vf fps=1 {save_path}/frame_%04d.png"
#             os.system(extracted_command)
           



def extract_frames(video_file):
    save_path = os.path.join('extracted_frames', '/'.join(os.path.splitext(video_file)[0].split('/')[1:]))
    os.makedirs(save_path, exist_ok=True)
    # if os.path.exists(save_path): 
    #     return

    extracted_command = f"ffmpeg -hwaccel cuda -i {video_file} -vf fps=1 {save_path}/frame_%04d.png"    # use gpu to speed up
    #extracted_command = f"ffmpeg -i {video_file} -vf fps=1 {save_path}/frame_%04d.png"     # cpu calculation. much slower.
    os.system(extracted_command)

all_video_files = glob('Activity_Videos/*') # for Video-ChatGPT
all_video_files += glob('sharegemini-core100k_video/*/*') # for ShareGemini

with ThreadPoolExecutor(max_workers=64) as executor: 
    futures = [executor.submit(extract_frames, video_file) for video_file in all_video_files]

    
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result() 