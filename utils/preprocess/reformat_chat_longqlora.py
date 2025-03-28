import json
def read_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


orig_data = read_json('LongQLoRA-format-10k.json')

new_data = []

for idx, item in enumerate(orig_data):
    convs = []
    cur_q = '<video>\n' + item['question']
    cur_a = item['answer']
    
    if 'Long Answer: ' in cur_a:
        # cur_a = cur_a.split('Long Answer: ')[0]
        # cur_a = cur_a.split('Answer: ')[1].strip()
        
        tmp_ans = item['answer'].strip().split('Long Answer: ')[0]
        short_ans = tmp_ans.split('Answer: ')[1].strip()
        doc_id = item['answer'].strip().split('Long Answer: ')[1].split('Gold Document ID: ')[1].strip()
        #cur_q += '\nAnswer the question using a single word or phrase.'
        cur_q += '\nAnswer the question in short and give the corresponding document ID.'
        cur_a = ""
        cur_a += short_ans + '\n'
        cur_a += "Document ID: " + doc_id

    convs.append(
        {
            'from': 'human',
            'value': cur_q
        }
    )
    convs.append(
        {
            'from': 'gpt',
            'value': cur_a
        }
    )

    new_data.append(
        {
            "conversations": convs,
            "video": f"LongQLoRA_{idx:05d}"
        }
    )

write_json("longqlora_chat_fake_vid.json", new_data)