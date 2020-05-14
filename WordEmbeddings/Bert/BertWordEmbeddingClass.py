#!/usr/bin/env python
# coding: utf-8

# In[22]:


from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
class bertWordEmbedding():
    def __init__(self):
        print('build model...')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased') 
        print('build end')
    def word2vec(self,text):
        print('analyzing...')
        text = '[SEP] '+text+' [CLS]'
        
        tokenizedtext = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenizedtext)
        tokens_tensor = torch.tensor([indexed_tokens])
         
        segments_tensors = torch.tensor([1]*len([tokenizedtext]))
        
        with torch.no_grad():
            encoded_layers,_= self.model(tokens_tensor,segments_tensors)
        
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings,dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        
        #sum last 4
        token_vec_sum = []
        for token in token_embeddings:
            sm = torch.sum(token[-4:],dim=0)
            token_vec_sum.append(sm)
        token_vec_sum = token_vec_sum[1:-1]
        print('wordEmbeddings result size is:{} words , size:{}*{} '.format(len(token_vec_sum),len(token_vec_sum),len(token_vec_sum[0])))  
        
        token_vecs = encoded_layers[11][0]
        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        print('sentenceEmbeddings result size is:{}'.format(len(sentence_embedding)))
        print('done')
        
        
        return (token_vec_sum,sentence_embedding)


# In[23]:


superman = bertWordEmbedding()


# In[24]:


a = superman.word2vec('Hello world')
a


# In[ ]:





# In[ ]:




