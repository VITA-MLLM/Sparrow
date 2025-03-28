import json
from tqdm import tqdm
def read_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

data = read_jsonl('booksum.jsonl')

print("# of samples:", len(data))

new_data = []

for sample in tqdm(data):
    #convs = []
    cur_q = sample['prompt']
    cur_a = sample['completion'].strip()
    # {text}\n\nQ: Can you write an appropriate summary of the above paragraphs?\nA:'
    context = cur_q.split('Q:')[0].strip()
    if len(context.split(' ')) < 3000:
        continue
    cur_q = cur_q.split('Q:')[1]
    cur_q = cur_q.split('A:')[0].strip()
    

    # convs.append(
    #     {
    #         'from': 'human',
    #         'value': cur_q
    #     }
    # )
    # convs.append(
    #     {
    #         'from': 'gpt',
    #         'value': cur_a
    #     }
    # )

    new_data.append(
        {
            'context': context,
            'question': cur_q,
            "answer": cur_a,
        }
    )

print("# of new samples:", len(new_data))
write_json("long-booksum.json", new_data)