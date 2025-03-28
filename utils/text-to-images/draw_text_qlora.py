import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import concurrent

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def split_text_by_word_limit_preserving_sentences(text, max_words_per_part):

    # 使用nltk的sent_tokenize来分割文本为句子
    sentences = sent_tokenize(text)
    parts = []
    current_part = []
    current_word_count = 0

    for sentence in sentences:
        # 计算当前句子的单词数
        word_count = len(sentence.split())
        # 检查是否添加这个句子会超过单词数限制
        if current_word_count + word_count > max_words_per_part:
            # 如果当前部分已经有内容，保存它并开始一个新的部分
            if current_part:
                parts.append(' '.join(current_part))
                current_part = [sentence]
                current_word_count = word_count
            else:
                # 如果当前部分为空（极长的句子），直接添加此句子
                parts.append(sentence)
                current_word_count = 0
        else:
            # 添加句子到当前部分
            current_part.append(sentence)
            current_word_count += word_count

    # 添加最后一部分（如果有）
    if current_part:
        parts.append(' '.join(current_part))

    return parts


def split_text_by_word_limit(text, max_words_per_part=90):
    # 使用nltk的word_tokenize来分割文本为单词
    words = word_tokenize(text)
    parts = []
    current_part = []
    current_word_count = 0

    for word in words:
        # 检查添加这个单词是否会超过单词数限制
        if current_word_count + 1 > max_words_per_part:
            # 如果会超过，保存当前部分并开始一个新的部分
            parts.append(' '.join(current_part))
            current_part = [word]
            current_word_count = 1
        else:
            # 否则，添加这个单词到当前部分
            current_part.append(word)
            current_word_count += 1

    # 添加最后一部分（如果有）
    if current_part:
        parts.append(' '.join(current_part))

    return parts


def split_text_into_parts(text, n_parts=128, max_words_per_part=90):

    # 使用nltk的sent_tokenize来分割文本为句子
    sentences = sent_tokenize(text)

    # 计算每部分应有的句子数量
    part_length = len(sentences) / n_parts
    parts = []
    current_part = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(word_tokenize(sentence))
        
        # 检查是否添加这个句子会超过单词数限制
        if current_word_count + sentence_word_count > max_words_per_part:
            # 如果会超过，保存当前部分并开始一个新的部分
            parts.append(' '.join(current_part))
            current_part = [sentence]
            current_word_count = sentence_word_count
        else:
            # 否则，添加这个句子到当前部分
            current_part.append(sentence)
            current_word_count += sentence_word_count

        # 检查是否达到平均句子数，这里使用ceil来确保至少分成n_parts部分
        if len(parts) + 1 < n_parts and len(current_part) >= round(part_length):
            parts.append(' '.join(current_part))
            current_part = []
            current_word_count = 0

    # 添加最后一部分（如果有）
    if current_part:
        parts.append(' '.join(current_part))

    return parts

def text_wrap(text, font, max_width):
    """
    将文本分行，使每行的宽度尽可能均匀。
    """
    words = text.split()
    lines = []
    current_line = []
    current_width = 0

    for word in words:
        word_width = font.getbbox(word + ' ')[2]
        if current_width + word_width <= max_width:
            current_line.append(word)
            current_width += word_width
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def create_image_with_text(text, font_path, font_size=24, image_size=(448, 448)):
    """
    创建一张包含文本的图片。
    """
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    # 文本分行处理
    lines = text_wrap(text, font, image.width - 40)  # 减去一些边距

    # 计算总文本块的高度
    total_height = sum(font.getbbox(line)[3] for line in lines)

    # 计算文本块开始的y坐标，使其垂直居中
    y = (image.height - total_height) / 2

    # 逐行绘制文本
    for line in lines:
        line_width, line_height = font.getbbox(line)[2:]
        x = 20  # 设置左边距
        draw.text((x, y), line, font=font, fill='black')
        y += line_height

    return image

def process_sample(sample, idx, base_output_dir, font_path, font_size):
    text = sample['context']
    output_dir = os.path.join(base_output_dir, f"LongQLoRA_{idx:05d}")
    os.makedirs(output_dir, exist_ok=True)
    save_text_to_images(text, output_dir, font_path, font_size)

def save_text_to_images(text, save_dir, font_path, font_size=24):
    """
    将文本分割成n_parts部分，并为每部分创建并保存一张图片。
    """
    #parts = split_text_into_parts(text, n_parts)
    parts = split_text_by_word_limit(text, 115)
    #parts = split_text_by_word_limit_preserving_sentences(text, 90)
    #images = []
    for i, part in enumerate(parts):
        image = create_image_with_text(part, font_path, font_size)
        image_path = os.path.join(save_dir, f'text_pic_{i:05d}.png')
        #os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)
    #     images.append(image_path)
    # return images

# 示例文本

def main():
    all_data = read_json("LongQLoRA-format-10k.json")
    base_output_dir = 'LongQLoRA-pics/'

    font_path = 'Arial.ttf'  # 你需要指定一个字体文件路径
    font_size = 20  # 你可以根据需要调整字体大小

    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(process_sample, sample, idx, base_output_dir, font_path, font_size) for idx, sample in enumerate(all_data)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass  # 这里可以处理每个future的结果或异常

    # for idx, sample in enumerate(tqdm(all_data)):
    #     text = sample['instruction']
    #     output_dir = os.path.join(base_output_dir, f"LongAlpaca_{idx:05d}")
    #     save_text_to_images(text, output_dir, font_path, font_size)


if __name__ == "__main__":
    main()
#text = all_data[0]['instruction'] 
#print("word count: ", len(text.split(' ')))

# text = ' '.join(text.split(' ')[:90])
# print("word count: ", len(text.split(' ')))
#tokenizer = AutoTokenizer.from_pretrained("/data/pandayin/ckpt/internvl-chat-4b")
#
## print(f"orig text token length: {len(tokenizer(text).input_ids)}"
## )
#font_path = 'arial.ttf'  # 你需要指定一个字体文件路径
#font_size = 24  # 你可以根据需要调整字体大小
#
## 分割文本
##parts = split_text_into_parts(text, 128)
#parts = split_text_by_word_limit(text, 70)
##parts = split_text_by_word_limit_preserving_sentences(text, 90)
#
#for idx, part in enumerate(parts):
#    print(f"Part {idx+1} of words: {len(part.split(' '))}")
#    #print(f"Part {idx+1} of length: {len(part)}")
#    print(f"Part {idx+1} of token length: {len(tokenizer(part).input_ids)}")
# # 打印结果
# for i, part in enumerate(parts):
#     print(f"Part {i+1}: {part}\n")  # 打印每部分的字符

#output_dir = "./tmp_ocr_out/"
#os.makedirs(output_dir, exist_ok=True)
#
#save_text_to_images(text, output_dir, 128, font_path, font_size)

