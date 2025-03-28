import json
from tqdm import tqdm
import random

'''
    The beginning of the context:
        Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).
'''
def read_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

data = read_jsonl('natural_questions_10_200_docs.jsonl')

print("# of samples:", len(data))

new_data = []
counter = []
for sample in tqdm(data):
    #convs = []
    cur_q = sample['prompt']
    #cur_a = sample['completion'].strip()
    # "Answer: a newsletter sent to an advertising firm's customers\nLong Answer: A common example of permission marketing is a newsletter sent to an advertising firm's customers . Such newsletters inform customers of upcoming events or promotions, or new products . In this type of advertising, a company that wants to send a newsletter to their customers may ask them at the point of purchase if they would like to receive the newsletter.\nGold Document ID: 24"
    # {text}\n\nQ: Can you write an appropriate summary of the above paragraphs?\nA:'
    context = cur_q.split('Question: ')[0].strip()
    cur_length = len(context.split(' '))
    if cur_length < 3000:
        continue
    else:
        counter.append(cur_length)
    

    cur_q = cur_q.split('Question: ')[1].strip().capitalize()
    if (not cur_q.endswith('?')) and (not cur_q.endswith('？')):
        cur_q += '?'
    
    cur_q += '\nAnswer the question in short and give the corresponding document ID.'

    tmp_ans = sample['completion'].strip().split('Long Answer: ')[0]
    short_ans = tmp_ans.split('Answer: ')[1].strip()
    doc_id = sample['completion'].strip().split('Long Answer: ')[1].split('Gold Document ID: ')[1].strip()
    
    cur_a = ""
    cur_a += short_ans + '\n'
    cur_a += "Document ID: " + doc_id
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

# import pdb;pdb.set_trace()
print("# of new samples:", len(new_data))
write_json("long-natq.json", new_data)