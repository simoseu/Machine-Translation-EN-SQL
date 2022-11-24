from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from sklearn.model_selection import train_test_split
import torch_optimizer as optim
import evaluate

# Load Spider dataset from local directory
db_path = ''
df = pd.read_json(db_path) 

# Delete useless columns
df.drop('query-split', axis=1, inplace=True)
df.drop('sql-original', axis=1, inplace=True)
df.drop('variables', axis=1, inplace=True)

df['sentences'] = df['sentences'].apply(lambda x: x[0]['text'])

# Splitting dataset into two subsets (Train and Test)
train, test = train_test_split(df, test_size = 0.20, random_state = 0)

# Print the dimension of two subsets
#print(train.shape)
#print(test.shape)

meteor = evaluate.load('meteor')

# Load tokenizer and model
model_path = ''
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# From df to list
train_sentences = train['sentences'].to_list()
test_sentences = test['sentences'].to_list()

train_query = train['sql'].to_list()
test_query = test['sql'].to_list()

# Flattening train query and test query
train_query_flatted = []
for lista in train_query:
    for stringa in lista:
        train_query_flatted.append(stringa)

test_query_flatted = []
for lista in test_query:
    for stringa in lista:
        test_query_flatted.append(stringa)


# Creating the training and test set tokenizing each element
train_set = list()
for e in range(len(train_sentences)):
  train_set.append({"input_ids": tokenizer(train_query_flatted[e], truncation = False, padding=True)["input_ids"], "labels": tokenizer(train_sentences[e], truncation = False, padding=True)["input_ids"]})


test_set = list()
for e in range(len(test_sentences)):
  test_set.append({"input_ids": tokenizer(test_query_flatted[e], truncation = False, padding=True)["input_ids"], "labels": tokenizer(test_sentences[e], truncation = False, padding=True)["input_ids"]})

#Evaluating the model
list_metric = []


for i in range(len(test_sentences)):
    preds = model.generate(tokenizer(test_query_flatted[i], return_tensors = "pt")["input_ids"], max_length = 150)
    predictions = [tokenizer.decode(t, skip_special_tokens=True) for t in preds]

    metrica = (meteor.compute(predictions = [predictions[0]], references = [test_sentences[i]]))
    list_metric.append(metrica['meteor'])
    print("Ciclo numero: ", i)

tot_score = sum(list_metric)
print("Total score: ", tot_score)

print("Average: ", tot_score/len(list_metric))