import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_k,embed_dim,num_heads,d_v=None):
        super(MultiHeadAttention,self).__init__()
        #assert embed_dim % num_heads == 0, "La dimensión del modelo debe ser divisible por el número de cabezas para hacer atención multicabeza"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = d_k
        self.key_query_dimension = d_k*num_heads
        if d_v is not None:
            self.d_v = d_v
            self.value_dimension = d_v*num_heads
        else:
            self.value_dimension=self.key_query_dimension
            self._d_v = d_k

        
        self.W_q = nn.Linear(embed_dim,self.key_query_dimension) # Query matrix of all heads
        self.W_k = nn.Linear(embed_dim,self.key_query_dimension) # Key matrix of all heads
        self.W_v = nn.Linear(embed_dim,self.value_dimension) # Value matrix  of all heads
        self.W_o = nn.Linear(self.value_dimension,embed_dim) # Matrix to resize multihead output
    def scaled_dot_product_attention(self,Q,K,V,mask=None):
        # Primero calculamos la atención de una palabra a otra:
        att_scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)

        # Para cuando se haga el Decoder se mete una máscara
        if mask is not None:
            #Aquellos valores de att_scores que coincidan con mask == 0 (broadcastable) son sustituidos por -1e9. 
            # De esta forma al aplicar softmax, al ser valores muy pequeños serán casi cero y no se les prestará atención.
            att_scores = att_scores.masked_fill(mask==0,-1e9) 
        att_prob = torch.softmax(att_scores,dim=-1)

        #Finalmente se obtienen los vectores de salida de la atención multicabeza al multiplicar por la matriz V
        output = torch.matmul(att_prob,V)
        return output
    def split_heads(self,x): 
        #Para poder hacer multihead attention tenemos que dividir las entradas del scaled_dot_attention_product. De esta forma, 
        #se puede paralelizar la computación (como dice en el paper original). 
        b_size,seq_length,_ = x.size()
        return x.view(b_size,seq_length,self.num_heads,-1).transpose(1,2)
    def concat_heads(self,x):
        b_size,_,seq_length,d_k = x.size()
        return x.transpose(1,2).contiguous().view(b_size,seq_length,self.value_dimension)
    def forward(self,Q,K,V,mask=None):
        #Multiplicación por las matrices de query,value y key. Luego, las salidas son proyectadas para llevar a cabo atención multicabeza.
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        #Atención multicabeza:
        att_output = self.scaled_dot_product_attention(Q,K,V,mask)
        output = self.W_o(self.concat_heads(att_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self,embed_dim,d_ff):
        super(PositionWiseFeedForward,self).__init__()
        self.ff1 = nn.Linear(embed_dim,d_ff)
        self.ff2 = nn.Linear(d_ff,embed_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.ff1(x))
        return self.ff2(x)   
    
class PositionalEncoding(nn.Module):
    def __init__(self,embed_dim,max_len):
        super(PositionalEncoding,self).__init__()
        posenc = torch.zeros(max_len,embed_dim)
        #unsqueeze to be broadcastable when multiplying with w_k dim((pos*w_k))=max_len x model_dim
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1) 
        w_k = torch.arange(0,embed_dim,2).float()*(1/(10000**(1/embed_dim)))

        posenc[:,0::2] = torch.sin(position*w_k) #los pares son el seno
        posenc[:,1::2] = torch.cos(position*w_k) #los impares son el coseno

        self.register_buffer('posenc',posenc.unsqueeze(0)) #Para que no se tenga en cuenta como parámetro entrenable
    def forward(self,x):
        #Se usa x.size() porque las entradas no tienen que ser todas de max_len. 
        return x + self.posenc[:,:x.size(1)] 
    

class EncoderLayer(nn.Module):
    def __init__(self,d_k,embed_dim,num_heads,d_ff,dropout=0.1,d_v=None):
        super(EncoderLayer,self).__init__()
        self.self_att = MultiHeadAttention(embed_dim=embed_dim,d_k=d_k,d_v=d_v,num_heads=num_heads)
        self.feed_forward = PositionWiseFeedForward(embed_dim,d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        att_output = self.self_att(x,x,x,mask)
        x = self.norm1(x+self.dropout(att_output)) #Residual connection + normalization
        ff_out = self.feed_forward(x)
        x = self.norm2(x+self.dropout(ff_out))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self,d_k,embed_dim,num_heads,d_ff,dropout=0.1,d_v=None):
        super(DecoderLayer,self).__init__()
        self.self_att = MultiHeadAttention(embed_dim=embed_dim,d_k=d_k,num_heads=num_heads,d_v=d_v)
        self.self_enc_att = MultiHeadAttention(embed_dim=embed_dim,d_k=d_k,num_heads=num_heads,d_v=d_v)
        self.feed_forward = PositionWiseFeedForward(embed_dim=embed_dim,d_ff=d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc_output,tgt_mask,input_mask=None):
        att_output = self.self_att(x,x,x,tgt_mask)
        x = self.norm1(x+self.dropout(att_output))
        enc_att_output = self.self_enc_att(x,enc_output,enc_output,input_mask)
        x = self.norm2(x+self.dropout(enc_att_output))
        ff_out = self.feed_forward(x)
        x = self.norm3(x+self.dropout(ff_out))
        return x