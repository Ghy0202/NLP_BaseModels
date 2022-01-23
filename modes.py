"""
模型部分：
    RNN
    LSTM
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# 自定义的类别引用
from utils.layers import DynamicLSTM


class LSTM(nn.Module):
    """
    标准LSTM
    """
    def __init__(self,embedding_matrix,opt):
        """
        :param embedding_matrix: 自定义词嵌入
        :param opt: 封装的参数
        """
        super(LSTM,self).__init__()
        self.embed=nn.Embedding.from_pretrained(torch.tensor(embedding_matrix,dtype=torch.float))
        self.lstm=DynamicLSTM(opt.embed_dim,opt.hidden_dim,num_layers=1,batch_first=True)
        #self.lstm=CustomLSTM(opt.embed_dim,opt.hidden_dim)
        #self.rnn=CustomRNN(opt.embed_dim,opt.hidden_dim)

        self.dense=nn.Linear(opt.hidden_dim,opt.polarities_dim)# 最后的输出


    def forward(self,inputs):
        text=inputs[0]
        x=self.embed(text)
        x_len=torch.sum(text!=0,dim=-1) # 每一个句子的长度

        _,(h_n,_)=self.lstm(x,x_len)
        #_, (h_n, _) = self.lstm(x)
        #h=self.rnn(x)

        #out=self.dense(h)
        #print("!",h_n[0].shape)
        out=self.dense(h_n[0])
        # 原本是h_n[0]

        return out                      # 输出分类结果

class MyLSTM(nn.Module):
    def __init__(self,embedding_matrix,opt):
        super(MyLSTM,self).__init__()
        self.embed=nn.Embedding.from_pretrained(torch.tensor(embedding_matrix,dtype=torch.float))
        '''LSTM的调用'''
        self.lstm=myLstm(opt.embed_dim,opt.hidden_dim)
        self.dense=nn.Linear(opt.hidden_dim,opt.polarities_dim)

    def forward(self,inputs):
        text = inputs[0]
        x = self.embed(text)

        _, (h_n, _) = self.lstm(x)

        #print("?",h_n.shape)
        out = self.dense(h_n)

        return out

class MyRNN(nn.Module):
    def __init__(self,embedding_matrix,opt):
        super(MyRNN,self).__init__()
        self.embed=nn.Embedding.from_pretrained(torch.tensor(embedding_matrix,dtype=torch.float))
        '''RNN的调用'''
        self.rnn=myrnn(opt.embed_dim,opt.hidden_dim)
        self.dense=nn.Linear(opt.hidden_dim,opt.polarities_dim)

    def forward(self,inputs):
        text=inputs[0]
        x=self.embed(text)
        h,y=self.rnn(x)
        #print(y.shape)
        out=self.dense(y)
        #print(out.shape)
        return out

class myrnn(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(myrnn,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        # RNN参数
        self.U=nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W=nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b=nn.Parameter(torch.Tensor(hidden_size))
        self.init_weights()
        # 这个函数捏，可以省略，但是收敛应该会差一些
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # 前向传播的过程
    def forward(self,inputs,init_state=None):
        batch_size,seq_size,_=inputs.size()
        # 初始化
        if init_state is None:
            h_t, c_t = (
                torch.zeros(batch_size, self.hidden_size).to(inputs.device),
                torch.zeros(batch_size, self.hidden_size).to(inputs.device)
            )
        else:
            h_t, c_t = init_state

        # 对于序列的时间步进行RNN step
        for t in range(seq_size):
            x_t = inputs[:, t, :]
            #print(x_t.shape,self.U.shape,h_t.shape)
            h_t=torch.sigmoid(x_t@self.U+h_t@self.W+self.b)
            y_t = torch.sigmoid(x_t@self.U + h_t@self.W + self.b)



        return h_t,y_t

# GRU模块的实现
class MyGRU(nn.Module):
    def __init__(self,embedding_matrix,opt):
        super(MyGRU,self).__init__()
        self.embed=nn.Embedding.from_pretrained(torch.tensor(opt.embed_dim,dtype=torch.float))
        self.mygru=mygru(opt.embed_dim,opt.hidden_dim)
        self.dense=nn.Linear(opt.hidden_dim,opt.polarities_dim)

    def forwards(self,inputs):
        text=inputs[0]
        x=self.embed(text)
        h,y=self.mygru(x)
        out=self.dense(y)
        return out

class mygru(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(mygru,self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size
        # GRU参数
        self.W_r=nn.Parameter(input_size,2*hidden_size)
        self.b_r=nn.Parameter(2*hidden_size)

        self.W_z = nn.Parameter(input_size, 2*hidden_size)
        self.b_z = nn.Parameter(2*hidden_size)

        self.W_h = nn.Parameter(input_size, hidden_size)
        self.b_h = nn.Parameter(hidden_size)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self,inputs):
        batch_size,seq_size,_=inputs.size()
        h_t, c_t = (
            torch.zeros(batch_size, self.hidden_size).to(inputs.device),
            torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        )
        # GRU 的 step
        for t in range(seq_size):
            x_t = inputs[:, t, :]
            h_t_1=h_t
            # 0-行 1-列
            # [bs,hidden][bs,hidden]
            z_t=torch.sigmoid(torch.cat((h_t_1,x_t),1)@self.W_z+self.b_z)
            r_t=torch.sigmoid(torch.cat((h_t_1,x_t),1)@self.W_r+self.b_r)
            y_t=torch.tanh(torch.cat((r_t*h_t_1,x_t),1)@self.W_h+self.b_h)
            h_t=(1-z_t)*h_t_1+z_t*y_t

        return h_t,y_t







class myLstm(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super(myLstm,self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t遗忘门
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t记忆门
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t 输出门
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()
    # 这个函数捏，可以省略，但是收敛应该会差一些
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.size()
        # 64 80 300
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device)
            )
        else:
            h_t, c_t = init_states
        for t in range(seq_sz):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)






