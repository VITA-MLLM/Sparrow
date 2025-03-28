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


orig_data = read_json('LongAlpaca-reformat-8k.json')

new_data = []

for idx, item in enumerate(orig_data):
    convs = []
    cur_q = '<video>\n' + item['question']
    cur_a = item['answer']
    
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
            "video": f"LongAlpaca_{idx:05d}"
        }
    )

write_json("longalpaca_chat_fake_vid.json", new_data)