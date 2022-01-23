import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils.layers import DynamicLSTM,Attention

class MYIAN(nn.Module):
    def __init__(self,embedding_matrix,opt):
        """
        初始化模型
        :param embedding_matrix:
        """
        super(MYIAN,self).__init__()
        self.opt=opt
        self.embed=nn.Embedding.from_pretrained(torch.tensor(embedding_matrix,dtype=torch.float))
        self.target_lstm=DynamicLSTM(opt.embed_dim,opt.hidden_dim,num_layers=1,batch_first=True)
        self.context_lstm=DynamicLSTM(opt.embed_dim,opt.hidden_dim,num_layers=1,batch_first=True)
        self.target_attention=Attention(opt.hidden_dim)
        self.context_attention=Attention(opt.hidden_dim)
        self.dense=nn.Linear(opt.hidden_dim*2,opt.polarities_dim)


    def forward(self,inputs):
        text,target=inputs[0],inputs[1]
        context_len=torch.sum(text!=0,dim=-1)
        target_len=torch.sum(target!=0,dim=-1)

        """lstm"""
        context=self.embed(text)
        target=self.embed(target)
        context_hidden_list,(_,_)=self.context_lstm(context,context_len)
        target_hidden_list,(_,_)=self.target_lstm(target,target_len)

        """pool"""
        target_len=torch.as_tensor(target_len,dtype=torch.float).to(self.opt.device)
        target_pool=torch.sum(target_hidden_list,dim=1)
        target_pool=torch.div(target_pool,target_len.view(target_len.size(0),1))

        context_len = torch.as_tensor(context_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(context_hidden_list, dim=1)
        context_pool = torch.div(context_pool, target_len.view(context_len.size(0), 1))

        """attention"""
        target_final,_=self.target_attention(target_hidden_list,context_pool)# target是k context_pool是q
        target_final=target_final.squeeze(dim=1)
        context_final,_=self.context_attention(context_hidden_list,target_pool)# context是k target是q--->对于特定的
        context_final=context_final.squeeze(dim=1)

        """合并"""
        x=torch.cat((target_final,context_final),dim=-1)

        """分类"""
        out=self.dense(x)

        return out










        pass
