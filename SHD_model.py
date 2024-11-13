from transformer import PositionalEncoding,EncoderLayer
import torch.nn as nn
import torch.nn.functional as F

class TransformerModelSHD(nn.Module):
    def __init__(self,embed_dim,num_heads,d_k,ff_dim,max_len,d_v=None):
        super(TransformerModelSHD,self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.transformerBlock = EncoderLayer(embed_dim=embed_dim,num_heads=num_heads,d_ff=ff_dim,d_k=d_k,d_v=d_v)
        self.posenc = PositionalEncoding(embed_dim=embed_dim,max_len=max_len)
        #self.embeding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim)
        self.fc1 = nn.Linear(embed_dim,100)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(100,20)
    def forward(self,x):
        x = self.posenc(x)
        x = self.transformerBlock(x)
        x = F.avg_pool1d(x.transpose(1,2),self.max_len).reshape(x.size(0),self.embed_dim)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = F.softmax(self.fc2(x),dim=-1)
        return output