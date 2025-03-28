import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import concurrent

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def split_text_by_word_limit_preserving_sentences(text, max_words_per_part):

    
    sentences = sent_tokenize(text)
    parts = []
    current_part = []
    current_word_count = 0

    for sentence in sentences:
        
        word_count = len(sentence.split())
        
        if current_word_count + word_count > max_words_per_part:
            
            if current_part:
                parts.append(' '.join(current_part))
                current_part = [sentence]
                current_word_count = word_count
            else:
                
                parts.append(sentence)
                current_word_count = 0
        else:
           
            current_part.append(sentence)
            current_word_count += word_count

   
    if current_part:
        parts.append(' '.join(current_part))

    return parts


def split_text_by_word_limit(text, max_words_per_part=90):
    
    words = word_tokenize(text)
    parts = []
    current_part = []
    current_word_count = 0

    for word in words:
        
        if current_word_count + 1 > max_words_per_part:
            
            parts.append(' '.join(current_part))
            current_part = [word]
            current_word_count = 1
        else:
            
            current_part.append(word)
            current_word_count += 1

    
    if current_part:
        parts.append(' '.join(current_part))

    return parts


def split_text_into_parts(text, n_parts=128, max_words_per_part=90):

    
    sentences = sent_tokenize(text)

    
    part_length = len(sentences) / n_parts
    parts = []
    current_part = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(word_tokenize(sentence))
        
        
        if current_word_count + sentence_word_count > max_words_per_part:
            
            parts.append(' '.join(current_part))
            current_part = [sentence]
            current_word_count = sentence_word_count
        else:
            
            current_part.append(sentence)
            current_word_count += sentence_word_count

        
        if len(parts) + 1 < n_parts and len(current_part) >= round(part_length):
            parts.append(' '.join(current_part))
            current_part = []
            current_word_count = 0

    
    if current_part:
        parts.append(' '.join(current_part))

    return parts

def text_wrap(text, font, max_width):
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
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    # split lines
    lines = text_wrap(text, font, image.width - 40)  # minus some margins

    # calculate height of text chunks
    total_height = sum(font.getbbox(line)[3] for line in lines)

    
    y = (image.height - total_height) / 2

    # draw text for each line
    for line in lines:
        line_width, line_height = font.getbbox(line)[2:]
        x = 20  # left margin
        draw.text((x, y), line, font=font, fill='black')
        y += line_height

    return image

def process_sample(sample, idx, base_output_dir, font_path, font_size):
    text = sample['context']
    output_dir = os.path.join(base_output_dir, f"LongAlpaca_{idx:05d}")
    os.makedirs(output_dir, exist_ok=True)
    save_text_to_images(text, output_dir, font_path, font_size)

def save_text_to_images(text, save_dir, font_path, font_size=24):
    """
    Split long text into {n_parts} chunks, and save for each chunk an image.
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



def main():
    all_data = read_json("LongAlpaca-reformat-8k.json")
    base_output_dir = 'LongAlpaca-pics/'

    font_path = 'Arial.ttf'  # Specify the path to the font.
    font_size = 20 

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sample, sample, idx, base_output_dir, font_path, font_size) for idx, sample in enumerate(all_data)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass  


if __name__ == "__main__":
    main()


