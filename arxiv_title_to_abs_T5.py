#!/usr/bin/env python
# coding: utf-8

# ## 🤗Transformers - Generating Articles from Paper's Abstracts using T5 Model
# This notebook uses T5 model - A Sequence to Sequence model fully capable to perform any text to text tasks. What does it mean - It means that T5 model can take any input text and convert it into any output text. Such Text to Text conversion is useful in NLP tasks like language translation, summarization etc.
# 
# In this notebook, we will take paper's abstracts as our input text and paper's title as output text and feed it to T5 model. So,let's dive in...
# 
# 

# We will install dependencies and work with latest stable pytorch 1.6
# 
# !pip uninstall torch torchvision -y
# 
# !pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# 
# !pip install -U transformers
# 
# !pip install -U simpletransformers  

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# In[2]:


import os, psutil  

def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))

# In[3]:


cpu_stats()

# In[4]:


import json

data_file = '../../arxiv_download/tools/arxiv-metadata-oai-snapshot.json'

def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line

# In[5]:


metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    print('Title: {}\n\nAbstract: {}\nRef: {}'.format(paper_dict.get('title'), paper_dict.get('abstract'), paper_dict.get('journal-ref')))
#     print(paper)
    break

# **We will take last 5 years ArXiv papers (2016-2021) due to Kaggle'c compute limits**

# In[6]:


titles = []
abstracts = []
years = []
metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    ref = paper_dict.get('journal-ref')
    try:
        year = int(ref[-4:]) 
        if 2010 < year < 2021:
            years.append(year)
            titles.append(paper_dict.get('title'))
            abstracts.append(paper_dict.get('abstract'))
    except:
        pass 

len(titles), len(abstracts), len(years)

# In[7]:


papers = pd.DataFrame({
    'title': titles,
    'abstract': abstracts,
    'year': years
})
papers.head()

# In[8]:


del titles, abstracts, years

# In[9]:


cpu_stats()

#  **We will use `simpletransformers` library to train a T5 model**

# In[10]:


import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# **Simpletransformers implementation of T5 model expects a data to be a dataframe with 3 columns:**
# `<prefix>, <input_text>, <target_text>`
# * `<prefix>`: A string indicating the task to perform. (E.g. "question", "stsb")
# * `<input_text>`: The input text sequence (we will use Paper's abstract as `input_text`  )
# * `<target_text`: The target sequence (we will use Paper's title as `output_text` )
#     
#     
#  You can read about the data format:  https://github.com/ThilinaRajapakse/simpletransformers#t5-transformer

# In[11]:


papers = papers[['title','abstract']]
papers.columns = ['target_text', 'input_text']
papers = papers.dropna()

# In[12]:


eval_df = papers.sample(frac=0.2, random_state=101)
train_df = papers.drop(eval_df.index)

# In[13]:


train_df.shape, eval_df.shape

# **We will training out T5 model with very bare minimum `num_train_epochs=4`, `train_batch_size=16` to  fit into Kaggle's compute limits**

# In[14]:


import logging

import pandas as pd
from simpletransformers.t5 import T5Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df['prefix'] = "summarize"
eval_df['prefix'] = "summarize"


model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "train_batch_size": 16,
    "num_train_epochs": 4,
}

# Create T5 Model
model = T5Model("t5", "t5-small", args=model_args, use_cuda=True)

# Train T5 Model on new task
model.train_model(train_df)

# Evaluate T5 Model on new task
results = model.eval_model(eval_df)

# Predict with trained T5 model
#print(model.predict(["convert: four"]))

# In[15]:


results

# ## And We're Done ! 
# **Let's see how our model performs in generating paper's titles**

# In[16]:


import random

document = []

for _ in range(250):
    data = dict()

    random_idx = random.randint(0, len(eval_df)-1)
    
    actual_abstract = eval_df.iloc[random_idx]['input_text']
    actual_title = eval_df.iloc[random_idx]['target_text']

    # Predict with trained T5 model
    predicted_title = model.predict(["summarize: " + actual_abstract])[0]

    print(f'Actual Title: {actual_title}')
    print(f'Predicted Title: {predicted_title}')
    print(f'Actual Abstract: {actual_abstract}')

    data['true_title'] = actual_title
    data['pred_title'] = predicted_title
    data['abstract'] = actual_abstract

    document.append(data)

import json
with open('result.json', 'w') as f:
    json.dump(document, f)


# In[ ]:



