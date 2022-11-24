from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch_optimizer as optim

# Load Spider dataset from local directory
save_location = ''
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

# Load tokenizer and model
model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Setting Training Arguments
batch_size = 16
optim='adamw_torch'
args = Seq2SeqTrainingArguments(
    f"fine_tuned",
    optim=optim,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=8,
    predict_with_generate=True,
)

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


# Training the model
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_set,
    eval_dataset=test_set,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()

#Saving the model
trainer.save_model(save_location)

#Testing the model with a given sentence
index_query = 10
preds = model.generate(tokenizer(test_query[index_query], return_tensors = "pt")["input_ids"], max_length = 100)

print([tokenizer.decode(t, skip_special_tokens=True) for t in preds])