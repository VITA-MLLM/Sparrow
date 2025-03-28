## Data Preparation

### Benchmark Preparation

The benchmark files should be organized as such in `benchmarks`.

```Shell
benchmarks
├── video-mme
├── mvbench
│   ├── video
│   └── mvbench.jsonl
└── TempCompass
│   ├── videos
│   └── TempCompass.jsonl
├── LongVideoBench
│   ├── videos
│   ├── subtitles
│   └── lvb_val.json
└── MLVU
    ├── video
    └── mlvu.jsonl
```
We have provided processed jsonl files of MVBench, TempCompass, and MLVU for easier reproduction.
For full files (such as videos), please refer to the official guidelines to (apply and) download.

### Video-MME
1. Follow the [instruction](https://github.com/BradyFU/Video-MME?tab=readme-ov-file#-dataset) to apply for the benchmark.
2. (Optional) Extract video frames to speed up the evaluation process (I/O for long videos can be time-consuming). You may refer to:
https://github.com/xjtupanda/T2Vid/blob/9bc94103f953ba2bfd9a267f652247b0765d9baa/utils/preprocess/extract_video_frames.py#L17-L30

### MVBench
1. Download the videos in [Link](https://huggingface.co/datasets/OpenGVLab/MVBench/tree/main/video).
2. Unzip all the files in the `video` folder.

### TempCompass
1. Download the videos at [tempcompass_videos.zip](https://huggingface.co/datasets/lmms-lab/TempCompass/blob/main/tempcompass_videos.zip).
2. Unzip the file and put all the videos in `videos` folder.

### LongVideoBench
1. Download the files at [Link](https://huggingface.co/datasets/longvideobench/LongVideoBench).
2. Follow the instructions and extract the tar files, including `videos.tar` and `subtitles.tar`.

### MLVU
1. Download the videos at [Link](https://huggingface.co/datasets/MLVU/MVLU).
2. Put all the videos in `video` folder.

## Evaluation

**Note:**
Please eval with a newer version of Transformers.
This is due to a incompatibility between the versions of MiniCPM-V (Training at ver. 4.40.0) and LLaMA-3.1 (For eval, Ver.4.46.1).
```shell
pip install transformers==4.46.1

Run the scripts `run_bench.sh` to evaluate on the three benchmarks, in `MiniCPM-V/eval/`.

Usage:
```Shell
cd MiniCPM-V/eval/
bash run_bench.sh {exp_name} {CKPT_file_path} {NUM_FRAMES}
```
For example, running `zero-shot` inference with the original image-LLM `Idefics3`, using `24` frames:
```Shell
bash run_bench.sh zero-shot openbmb/MiniCPM-Llama3-V-2_5 24
```

We provide fine-tuned weights in our [Hugging Face collection](https://huggingface.co/collections/xjtupanda/t2vid-673f104cdaf4ac3340b15964).
