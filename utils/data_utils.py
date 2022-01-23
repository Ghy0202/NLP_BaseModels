import os
import pickle
import numpy as np

import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

def parseXML(data_path):
    tree=ET.ElementTree(file=data_path)
    objs=list()
    for sentence in tree.getroot():
        obj=dict()
        for item in sentence:
            if item.tag=='text':
                obj['text']=item.text
            elif item.tag=='aspectTerms':
                obj['aspects']=list()
                for aspectTerm in item:
                    if aspectTerm.attrib['polarity']!='conflict':
                        obj['aspects'].append(aspectTerm.attrib)
        if 'aspects' in obj and len(obj['aspects']):
            objs.append(obj)
    return objs

def build_tokenizer(fnames,max_length,data_file):
    # 如果已经转化成二进制，直接读取，反之自己创建
    if os.path.exists(data_file):
        print('loading tokenizer:',data_file)
        tokenizer=pickle.load(open(data_file,'rb'))

    else:
        tokenizer=Tokenizer.from_files(fnames=fnames,max_length=max_length)
        pickle.dump(tokenizer,open(data_file,'wb'))

    return tokenizer



# 下面是Tokenizer的具体创建
class Tokenizer(object):
    '''将text转成index'''
    def __init__(self,vocab,max_length,lower):
        self.vocab=vocab
        self.max_length=max_length
        self.lower=lower

    @classmethod
    def from_files(cls,fnames,max_length,lower=True):
        corpus=set()
        for fname in fnames:
            for obj in parseXML(fname):
                text_raw=obj['text']
                if lower:
                    text_raw=text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw))
        return cls(vocab=Vocab(corpus,add_pad=True,add_unk=True),max_length=max_length,lower=lower)
    @staticmethod
    def pad_sequence(sequence,pad_id,maxlen,dtype='int64',padding='post',truncating='post'):
        x=(np.zeros(maxlen)+pad_id).astype(dtype)
        if truncating=='pre':
            trunc=sequence[-maxlen:]
        else:
            trunc=sequence[:maxlen]
        trunc=np.asarray(trunc,dtype=dtype)
        if padding=='post':
            x[:len(trunc)]=trunc
        else:
            x[-len(trunc):]=trunc
        return x

    @staticmethod
    def split_text(text):
        for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
            text=text.replace(ch," "+ch+" ")
        return text.strip().split()

    def position_sequence(self,text,start,end,reverse=False,padding='post',truncating='post'):
        text_left=Tokenizer.split_text(text[:start])
        text_aspect=Tokenizer.split_text(text[start:end])
        text_right=Tokenizer.split_text(text[end:])
        tag_left=[len(text_left)-i for i in range(len(text_left))]
        tag_aspect=[0 for i in range(len(text_aspect))]
        tag_right=[i+1 for i in range(len(text_right))]
        position_tag=tag_left+tag_aspect+tag_right
        if len(position_tag)==0:
            position_tag=[0]
        if reverse:
            position_tag.reverse()
        return Tokenizer.pad_sequence(position_tag,pad_id=0,maxlen=self.max_length,
                                      padding=padding,truncating =truncating)

    def text_to_sequence(self,text,reverse=False,padding='post',truncating='post'):
        if self.lower:
            text=text.lower()
        words=Tokenizer.split_text(text)
        sequence=[self.vocab.word_to_id(w) for w in words]
        if len(sequence)==0:
            sequence=[0]
        if reverse:
            sequence.reverse()

        return Tokenizer.pad_sequence(sequence,pad_id=self.vocab.pad_id,maxlen=self.max_length,padding=padding,truncating=truncating)


# 下面是Vocab的具体创建
class Vocab(object):
    '''数据集的词典'''
    def __init__(self,vocab_list,add_pad,add_unk):
        self._vocab_dict=dict()
        self._reverse_vocab_dict=dict()
        self._length=0
        if add_pad:
            self.pad_word='<pad>'
            self.pad_id=self._length
            self._length+=1
            self._vocab_dict[self.pad_word]=self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w]=self._length
            self._length+=1

        for w,i in self._vocab_dict.items():
            self._reverse_vocab_dict[i]=w

    def word_to_id(self,word):
        if hasattr(self,'unk_id'):
            return self._vocab_dict.get(word,self.unk_id)
        return self._vocab_dict[word]


    def id_to_word(self,id):
        if hasattr(self,'unk_word'):
            return self._reverse_vocab_dict.get(id,self.unk_word)
        return self._reverse_vocab_dict[id]

    def has_word(self,word):
        return word in self._vocab_dict


    def __len__(self):
        return self._length

def _load_wordvec(data_path,vocab=None):
    with open(data_path,'r',encoding='utf-8',newline='\n',errors='ignore') as f:
        word_vec=dict()
        for line in f:
            tokens=line.rstrip().split()
            if tokens[0]=='<pad>' or tokens[0]=='<unk>':
                continue
            if vocab is None or vocab.has_word(tokens[0]):
                word_vec[tokens[0]]=np.asarray(tokens[1:],dtype='float32')
        return word_vec


def build_embedding_matrics(vocab,embed_dim,data_file,glove=True):
    if os.path.exists(data_file):
        print('loading embedding matrix:',data_file)
        embedding_matrix=pickle.load(open(data_file,'rb'))
    else:
        print('loading word vectors……')

        embedding_matrix=np.zeros((len(vocab),embed_dim))

        fname = './glove/glove.twitter.27B/glove.twitter.27B.'+str(embed_dim)+'d.txt' if embed_dim != 300 else './glove/glove.42B.300d.txt'

        word_vec=_load_wordvec(fname,vocab)
        for i in range(len(vocab)):
            vec=word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i]=vec
        pickle.dump(embedding_matrix,open(data_file,'wb'))
    return embedding_matrix

class SentenceDataset(Dataset):
    def __init__(self,fname,tokenizer,target_dim):
        data=list()
        polarity_dict={'positive':0,'negative':1,'neutral':2}
        for obj in parseXML(fname):
            t=obj['text']
            text=tokenizer.text_to_sequence(obj['text'])

            for aspect in obj['aspects']:
                if target_dim==2 and aspect['polarity']=='neutral':
                    continue
                aspect_term=tokenizer.text_to_sequence(aspect['term'])
                position=tokenizer.position_sequence(obj['text'],int(aspect['from']),int(aspect['to']))
                polarity=polarity_dict[aspect['polarity']]
                left_with_aspect_index=tokenizer.text_to_sequence(t+" "+aspect['term'])
                right_with_aspect_index=tokenizer.text_to_sequence(aspect['term']+" "+t,reverse=True)

                data.append({'text':text,'aspect':aspect_term,'position':position,'polarity':polarity})
        self._data=data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)



