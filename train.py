import os
import torch
import torch.nn as nn
import argparse
from sklearn import metrics
from torch.utils.data import DataLoader
import json
import pandas as pd
import csv
# 自己写的轮子
from modes import LSTM,MyLSTM,MyRNN,MyGRU
from utils.data_utils import  SentenceDataset,build_tokenizer, build_embedding_matrics
from IAN import MYIAN
# 绘图
import matplotlib.pyplot as plt
import datetime
# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号

class Instructor:

    def __init__(self,opt):
        self.opt=opt
        tokenizer=build_tokenizer(
            fnames=[opt.dataset_file['train'],opt.dataset_file['test']],
            max_length=opt.max_length,
            data_file='{0}_tokenizer.dat'.format(opt.dataset)
        )
        embedding_matrix=build_embedding_matrics(
            vocab=tokenizer.vocab,
            embed_dim=opt.embed_dim,
            data_file='{}_{}_embedding_matrix.dat'.format(str(opt.embed_dim),opt.dataset),
            glove=True
        )
        trainset=SentenceDataset(opt.dataset_file['train'],tokenizer,target_dim=opt.polarities_dim)
        testset=SentenceDataset(opt.dataset_file['test'],tokenizer,target_dim=opt.polarities_dim)
        self.train_dataloader=DataLoader(dataset=trainset,batch_size=opt.batch_size,shuffle=True)
        self.test_dataloader=DataLoader(dataset=testset,batch_size=opt.batch_size,shuffle=True)
        self.model=opt.model_class(embedding_matrix,opt).to(opt.device)
        # 关于cuda的部分如何去写
        if opt.device.type=='cuda':
            print('cuda memory allocated:',torch.cuda.memory_allocated(self.opt.device.index))
        # 实验参数
        self._print_args()

    def _print_args(self):
        # 这边套用原项目
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape)>1:
                    self.opt.initializer(p)
                else:
                    stdv=1./(p.shape[0]**0.5)# TODO：这一步是为了做什么？梯度清零？
                    torch.nn.init.uniform_(p,a=-stdv,b=stdv)

    def run(self,repeats=1):
        criterion=nn.CrossEntropyLoss()
        _params=filter(lambda p: p.requires_grad,self.model.parameters())
        optimizer=self.opt.optimizer(_params,lr=self.opt.learning_rate,weight_decay=self.opt.l2reg)
        # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.78)
        #TODO:momentum的设置
        max_test_acc_over_all=0
        max_f1_over_all=0
        for i in range(repeats):
            print('repeat:',i)
            self._reset_params()
            max_test_acc,max_f1,losses=self._train(criterion,optimizer,max_test_acc_over_all)
            print('max_test_acc:{0} ,max_f1:{1}'.format(max_test_acc,max_f1_over_all))
            max_test_acc_over_all=max(max_test_acc_over_all,max_test_acc)
            max_f1_over_all=max(max_f1_over_all,max_f1)
            print("*"*50)
        print('max_test_acc_overall:',max_test_acc_over_all)
        print('max_f1_overall:',max_f1_over_all)
        # 绘制损失函数曲线
        plt.plot(losses,color='r')
        plt.legend(labels=['loss'],loc='best')
        plt.title('Scratch_Loss')
        plt.show()
        return max_test_acc_over_all,max_f1_over_all

        pass

    def _train(self,criterion,optimizer,max_test_acc_overall):
        max_test_acc=0
        max_f1=0
        global_step=0
        loss_list=[]
        for epoch in range(self.opt.num_epoch):
            print('>'*50)
            print('epoch:',epoch)
            n_correct,n_total=0,0
            for i_batch,sample_batch in enumerate(self.train_dataloader):
                global_step+=1
                self.model.train()
                optimizer.zero_grad()
                # 模型训练阶段
                inputs=[sample_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs=self.model(inputs)
                #torch.Tensor shape[64,3]
                #print(outputs)
                #print(outputs.shape)
                targets=sample_batch['polarity'].to(self.opt.device)

                # 损失函数计算以及反向更新参数

                loss=criterion(outputs,targets)
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()

                if global_step%self.opt.log_step==1:
                    n_correct+=(torch.argmax(outputs,-1)==targets).sum().item()# 预测正确的个数
                    n_total+=len(outputs)
                    train_acc=n_correct/n_total
                    test_acc,f1=self._evaluate()# 验证测试集上的准确率
                    if test_acc >max_test_acc:
                        max_test_acc=test_acc
                        if test_acc>max_test_acc_overall:
                            # 如果是全局最大的acc保存模型相关参数
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = './state_dict/{0}_{1}_{2}class_acc{3:.4f}'.format(self.opt.model_name, self.opt.dataset, self.opt.polarities_dim, test_acc)
                            torch.save(self.model.state_dict(), path)
                            print('model saved:', path)

                    if f1>max_f1:
                        max_f1=f1
                    print('loss:{:.4f},acc:{:.4f},test_acc:{:.4f},f1:{:.4f}'.format(loss.item(),train_acc,test_acc,f1))
        return max_test_acc,max_f1,loss_list

    def _evaluate(self):
        # 利用训练的模型进行评估
        self.model.eval()
        n_test_correct,n_test_total=0,0
        t_targets_all,t_outputs_all=None,None
        with torch.no_grad():
            for t_batch,t_sample_batch in enumerate(self.test_dataloader):
                t_inputs=[t_sample_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets=t_sample_batch['polarity'].to(self.opt.device)
                t_outputs=self.model(t_inputs)

                # 计算正确率
                n_test_correct+=(torch.argmax(t_outputs,-1)==t_targets).sum().item()
                n_test_total+=len(t_outputs)

                t_targets_all=torch.cat((t_targets_all,t_targets),dim=0) if t_targets_all is not None else t_targets
                t_outputs_all=torch.cat((t_outputs_all,t_outputs),dim=0) if t_outputs_all is not None else t_outputs
            test_acc=n_test_correct/n_test_total
            f1=metrics.f1_score(t_targets_all.cpu(),torch.argmax(t_outputs_all,-1).cpu(),labels=[0,1,2],average='macro')
            return test_acc,f1




def main():
    model_classes={
        'lstm':LSTM,
        'my_lstm':MyLSTM,
        'my_rnn':MyRNN,
        'my_gru':MyGRU,
        'my_ian':MYIAN
    }


    dataset_files={
        'restaurant': {
            'train': './datasets/Restaurants_Train.xml',
            'test': './datasets/Restaurants_Test.xml'
        },
        'laptop': {
            'train': './datasets/Laptops_Train.xml',
            'test': './datasets/Laptops_Test.xml'
        }
    }

    input_colses={
        'lstm':['text'],
        'my_lstm': ['text'],
        'my_rnn':['text'],
        'my_gru':['text'],
        'my_ian':['text','aspect']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    # 实验参数
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='lstm', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--position_dim', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int, help='2, 3')
    parser.add_argument('--max_length', default=80, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--repeats', default=1, type=int)
    parser.add_argument('--max_test_acc', default=0.0, type=float)
    parser.add_argument('--max_test_f1', default=0.0, type=float)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(
        opt.device)


    ins = Instructor(opt)
    acc_,f1_=ins.run(opt.repeats)
    opt.max_test_f1=f1_
    opt.max_test_acc=acc_
    #TODO：这边需要添加实验结果记录
    my_log(opt)



def my_log(opt):
    file = './log/logging.csv'

    '''表头'''
    # namelist4 = ['model_name', 'dataset', 'optimizer', 'initializer', 'lr',
    #              'dropout','l2reg','num_epoch','batch_size','log_step',
    #              'embed_dim','hidden_dim','position_dim','max_length','device',
    #              'repeats','max_test_acc','max_f1'
    #              ]
    # df = pd.read_csv(file, header=None, names=namelist4)
    # df.to_csv(file, index=False)

    with open(file,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [opt.model_name,opt.dataset,opt.optimizer,opt.initializer,opt.learning_rate,
                    opt.dropout,opt.l2reg,opt.num_epoch,opt.batch_size,opt.log_step,
                    opt.embed_dim,opt.hidden_dim,opt.position_dim,opt.max_length,opt.device,
                    opt.repeats,opt.max_test_acc,opt.max_test_f1]
        csv_write.writerow(data_row)





if __name__=='__main__':
    main()



    """
    实验结果记录：
    Laptop:
    adamax
    max_test_acc_overall: 0.6974921630094044
    max_f1_overall: 0.6228886127894268
    
    asgd:提高学习率会好一些，收敛的不算快
    max_test_acc_overall: 0.6285266457680251
    max_f1_overall: 0.5282686012570744

    adagrad
    max_test_acc_overall: 0.6974921630094044
    max_f1_overall: 0.6500044697302954
    40
    max_test_acc_overall: 0.7053291536050157
    max_f1_overall: 0.6416808036805105

    """


