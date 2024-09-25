'''
Author: Yasmin Qin
Date: 2024-05-27 10:25:21
LastEditTime: 2024-06-18 10:12:44
'''
import torch
from torch.nn import Module
from torch import nn
import math


class TransformerBlock(Module):
    def __init__(self,token_num,d_model,num_head,d_diff,drop_out) -> None:
        super(TransformerBlock,self).__init__()
        self.multi_attention=MultiHeadAttention(token_num,d_model,num_head)
        self.ln1=nn.LayerNorm(d_model)
        self.fnn=FFN(d_model, d_diff, drop_out)
        self.ln2=nn.LayerNorm(d_model)
        
    def forward(self,x):
        #(B,num_patches+1,emded_dim)
        x=self.ln1(self.multi_attention(x)+x) #(B,num_patches+1,emded_dim)
        x=self.ln2(self.fnn(x)+x)#(B,num_patches+1,emded_dim)
        return x

class ViT(Module):
    def __init__(self,image_size,patch_size,d_model,num_head,d_diff,drop_out) -> None:
        super(ViT,self).__init__()
        self.patch_emded=PatchedEmbedding(image_size,patch_size,3,d_model,drop_out)
        token_num=(image_size//patch_size)*(image_size//patch_size)+1
        self.module_list=nn.ModuleList()
        for i in range(4):
            self.module_list.append(TransformerBlock(token_num,d_model,num_head,d_diff,drop_out))
        
        self._init_weight()
    
    def forward(self,x):
        x=self.patch_emded(x)
        for module in self.module_list:
            x=module(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class FFN(Module):
    def __init__(self,d_model,d_diff,drop_out) -> None:
        super(FFN,self).__init__()
        
        self.ffn=nn.Sequential(nn.Linear(d_model,d_diff),
                               nn.ReLU(),
                               nn.Dropout(drop_out),
                               nn.Linear(d_diff,d_model))
        
    def forward(self,x):
        
        x=self.ffn(x)#(B,num_patches+1,emded_dim)
        return x

class MultiHeadAttention(Module):
    def __init__(self,token_num,d_model,num_heads) -> None: #d_model 为embed_dim
        super(MultiHeadAttention,self).__init__()
        
        self.d_model=d_model
        self.depth = d_model//num_heads
        self.num_heads= num_heads
        self.query_linear=nn.Linear(d_model,d_model)
        self.key_linear=nn.Linear(d_model,d_model)
        self.value_linear=nn.Linear(d_model,d_model)
        self.out_linear=nn.Linear(d_model,d_model)
        self.soft_max=nn.Softmax(dim=-1)
        self.token_num=token_num
    
    def split_head(self,x):
        x=x.view(-1,self.token_num,self.num_heads,self.depth)
        return x.permute(0,2,1,3)
        
    def forward(self,x):
        
        query = self.query_linear(x)#(B,num_patches+1,emded_dim)
        key = self.key_linear(x)
        value = self.value_linear(x)
        query=self.split_head(query)#(B,num_heads,num_patches+1,depth)
        key=self.split_head(key)
        value=self.split_head(value)
        attention_w=self.soft_max(torch.matmul(query,key.transpose(3,2))/self.depth**0.5)
        out=torch.matmul(attention_w,value)#(B,num_heads,num_patches+1,depth)
        out = out.permute(0,2,1,3)#(B,num_patches+1,num_heads,depth)
        out =  out.reshape(-1,self.token_num,self.d_model) #(B,num_patches+1,embed_dim)
        out = self.out_linear(out)
        return out


class ShiftWindowMultiHeadAttention(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        return
    
    def window_partion(self,x,window_size):
        
        x=x.reshape()
        return
    

class PatchedEmbedding(Module):
    def __init__(self,image_size,patch_size,in_channel,emded_dim,drop_out) -> None: #emded_dim=patch_size*patch_size*in_channel
        super(PatchedEmbedding,self).__init__()
        num_patches=(image_size//patch_size)*(image_size//patch_size)
        self.patch_embedding=nn.Conv2d(in_channel,emded_dim,patch_size,stride=patch_size) #此为切割原图为patch输出(B,      emded_dim,          image_size//patch_size,image_size//patch_size)
        self.class_token=nn.parameter.Parameter(torch.zeros((1,1,emded_dim)))                                  #batch   每个patch的feature   后面的两维分别是patch划分下宽高下得到的patch的个数，相乘为总个数
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches+1, emded_dim)) 
        self.drop_out = nn.Dropout(drop_out)
        
    def forward(self,x):
        ct=self.class_token.expand((x.shape[0],-1,-1))
        x = self.patch_embedding(x)      #(B,emded_dim,num_patch,num_patch)
        x = x.flatten(2)                 #(B,emded_dim,num_patches)
        x = x.transpose(2,1)           #(B,num_patches,emded_dim)
        x = torch.concat([ct,x],axis=1)  #(B,num_patches+1,emded_dim)
        x = x+self.pos_emb #(B,num_patches+1,emded_dim)
        x = self.drop_out(x)#(B,num_patches+1,emded_dim)
        return x

if __name__ == '__main__':
    vit = ViT(224,16,768,8,1024,0.1)
    print(vit)
    img=torch.randn((1,3,224,224))
    output=vit(img)
    print(output.shape)