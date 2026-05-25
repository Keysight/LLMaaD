import json, csv
from pathlib import Path
PATH = Path(__file__).resolve().parent / "data"

def load_json(data_path):
    with data_path.open("r") as f:
        data = json.load(f)
    return data

def load_dataset(data_set):
    #This dataset contains Hamrful Question w.r.t custom generated same intent harmful genereated sentences use system prompt from prompt_reshaping/harmful_sentence_gen/generator.py to generate more
    if data_set == "dataset2":    
        data_path = PATH / "harmful_sentences_gen_latest.json"
        data = load_json(data_path)
        questions = []
        harmful_sentences = []
        for key, value in data.items():
            questions.append(key)
            harmful_sentences.append(value)
        return questions, harmful_sentences
    
    elif data_set == "dataset1":    
        list1 = []
        list2 = []
        data_path = PATH / "harmful_behaviors.csv"
        with data_path.open(newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:  # make sure both columns exist
                    list1.append(row[0])
                    list2.append(row[1])
        return list1, list2
    
    else:
        data_path = PATH / data_set
        data = load_json(data_path)
        questions = []
        harmful_sentences = []
        for key, value in data.items():
            questions.append(key)
            harmful_sentences.append(value)
        return questions, harmful_sentences

