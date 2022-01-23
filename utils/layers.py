import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class DynamicLSTM(nn.Module):
    """支持动态长度的LSTM"""
    def __init__(self,input_size,hidden_size,num_layers=1,bias=True,batch_first=True,dropout=0,
                 bidirectional=False,only_use_last_hidden_state=False,rnn_type='LSTM'):
        super(DynamicLSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.bias=bias
        self.batch_first=batch_first
        self.dropout=dropout
        self.bidirectional=bidirectional
        self.only_use_last_hidden_state=only_use_last_hidden_state
        self.rnn_type=rnn_type
        self.h_t_1 = None
        self.c_t_1 = None

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
            # print("手写LSTM")
            # self.initParams=None
            # self.RNN=ScratchLSTM(input_size=input_size,hidden_size=hidden_size)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self,x,x_len):
        '''
                sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
                '''
        if self.rnn_type=='LSTMscratch':
            out, (h_t, c_t) = self.RNN(x,self.h_t_1,self.c_t_1)
            self.h_t_1=h_t
            self.c_t_1=c_t
            return out, (h_t, c_t)
        else:
            '''sort'''
            x_sort_idx = torch.sort(x_len, descending=True)[1].long()
            x_unsort_idx = torch.sort(x_sort_idx)[1].long()
            x_len = x_len[x_sort_idx]

            x = x[x_sort_idx]
            '''pack'''
            x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
            ''' process '''
            if self.rnn_type == 'LSTM':
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, None)
                ct = None
            '''unsort'''
            ht = ht[:, x_unsort_idx]
            if self.only_use_last_hidden_state:
                return ht
            else:
                out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
                if self.batch_first:
                    out = out[x_unsort_idx]
                else:
                    out = out[:, x_unsort_idx]
                if self.rnn_type == 'LSTM':
                    ct = ct[:, x_unsort_idx]
                return out, (ht, ct)


# Attention机制的理解
class Attention(nn.Module):
    def __init__(self,embed_dim,hidden_dim=None,out_dim=None,n_head=1,score_function='mlp'):
        super(Attention,self).__init__()
        if hidden_dim is None:
            hidden_dim=embed_dim//n_head
        if out_dim is None:
            out_dim=embed_dim
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.n_head=n_head
        self.score_function=score_function
        self.W_k=nn.Linear(embed_dim,n_head*hidden_dim)
        self.W_q=nn.Linear(embed_dim,n_head*hidden_dim)
        self.proj=nn.Linear(n_head*hidden_dim,out_dim)
        if score_function=='mlp':
            self.weight=nn.Parameter(torch.Tensor(2*hidden_dim))

        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self,k,q):
        """
        前馈过程
        :param k: key原来的信息
        :param q: query咨询的问题
        :return:
        """
        # 增加维度的
        if len(q.shape)==2:
            q=torch.unsqueeze(q,dim=1)#eg:[32, 200]->[32, 1, 200]
        if len(k.shape)==2:
            k=torch.unsqueeze(k,dim=1)#eg:[32, 8, 200]没变
        # TODO:不了解
        mb_size=k.shape[0]#batch_size
        k_len=k.shape[1]# 键值的长度
        q_len=q.shape[1]# query的长度
        kx=self.W_k(k).view(mb_size,k_len,self.n_head,self.hidden_dim)# 转变形状
        kx=kx.permute(2,0,1,3).contiguous().view(-1,k_len,self.hidden_dim)# permute是改变维度的顺序
        qx = self.W_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)# permu
        if self.score_function=='bi_linear':
            qw=torch.matmul(qx,self.weight)
            kt=kx.permute(0,2,1)
            score=torch.bmm(qw,kt)# 一种乘法
            # TODO:"mlp"好像才是这篇文章里的方法
        elif self.score_function=="mlp":
            kxx=torch.unsqueeze(kx,dim=1).expand(-1,q_len,-1,-1)
            qxx=torch.unsqueeze(qx,dim=2).expand(-1,-1,k_len,-1)
            kq=torch.cat((kxx,qxx),dim=-1)
            score=torch.tanh(torch.matmul(kq,self.weight))
        score=F.softmax(score,dim=-1)
        output=torch.bmm(score,kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        return output,score











