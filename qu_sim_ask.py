import pandas as pd
import Levenshtein
import numpy as np
import nltk
#nltk.download()
from nltk.corpus import wordnet

#test=pd.read_csv('test.csv').fillna('')
train=pd.read_csv('train.csv',nrows=20000).fillna('')

train.question1=train.question1.str.replace('?','').str.upper()
train.question2=train.question2.str.replace('?','').str.upper()

#test.question1=train.question1.str.replace('?','').str.upper()
#test.question2=train.question2.str.replace('?','').str.upper()

def words_intersection(x,y):
    if (x=='') or (y==''):
        return 0.0
    
    else:
        shorter=min([len(x),len(y)])
        common=set(x.split(' ')).intersection(y.split(' '))
        return len(common)/float(shorter)

        

def to_words_tags(s):
    text=nltk.word_tokenize(s)
    return ' '.join([x[1] for x in nltk.pos_tag(text)])

def tags_distance(q1,q2):
    q1 = to_words_tags(q1)
    q2 = to_words_tags(q2)
    return Levenshtein.ratio(q1,q2)


train['intersection'] = train.apply(lambda x: [words_intersection(x['question1'],x['question2']),
Levenshtein.ratio(x['question1'],x['question2']), tags_distance(x['question1'],x['question2'])], axis=1) # normalized intersection
#train['similarity'] = train.apply(lambda x: Levenshtein.ratio(x['question1'],x['question2']), axis=1) # normalized Levenshtein
#train['tag_similarity'] = train.apply(lambda x: tags_distance(x['question1'],x['question2']), axis=1) # Levenshtein distance of tagged words



