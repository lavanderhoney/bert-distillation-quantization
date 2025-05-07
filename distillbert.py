import torch
import torch.nn as nn

class DistilBERTEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size, #inp for token embedding
                 max_len, #inp for position embedding
                 embed_dim,#embedded vector size. or the encoder's hidden size
                 dropout=0.1 #dropout rate
                ):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim).cuda()
        self.pos_embed = nn.Embedding(max_len, embed_dim).cuda()
#         self.seg_embed = nn.Embeddings(n_segs, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        #hardcode the initial input postion vector
        self.pos_inp = torch.tensor([i for i in range(max_len)],).cuda()
        
        #provide the sequence and segment vector.
        #No need to give positon vector as it initialized as an attribute 
    def forward(self, seq):
        embed_val = self.tok_embed(seq).to(torch.device("cuda")) + self.pos_embed(self.pos_inp).cuda()
        return self.drop(embed_val).to(torch.device("cuda")) #return the embedding vector of the input sequence
    
class MLPLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_layer,
                 dropout
                ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, embed_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(self.dropout(x))
        
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len,
                 embed_dim,
                 attn_heads,
                 n_layers,
                 dropout
                ):
        super().__init__()
        self.embedding = DistilBERTEmbedding(vocab_size, max_len, embed_dim, dropout).cuda()
        self.multi_head = nn.MultiheadAttention(embed_dim, #input feature size for the transformer block
                                                attn_heads,#no of heads
                                                dropout,
                                                batch_first=True
                                               ).cuda()
        self.mlp = MLPLayer(embed_dim, 4*embed_dim, dropout)
        self.layernorm = nn.LayerNorm(embed_dim).cuda()

    def forward(self, embeddings, mask):
        # print("embed shape: ", embeddings.shape)
        # print("mask shape: ", mask.shape)
        embeddings = embeddings.squeeze(1).cuda()
        mask = mask.squeeze(1).cuda()
        interacted, attn_weights = self.multi_head(embeddings, embeddings, embeddings, key_padding_mask=mask)
        interacted = self.layernorm(embeddings + interacted) #residual connection
        mlp_out = self.mlp(interacted)
        encoded = self.layernorm(mlp_out + interacted)
        return encoded
    
class DistilBERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    DistillBERT from original paper
    """
    def __init__(self,
                 vocab_size,
                 max_len,
                 n_segs,
                 embed_dim, 
                 n_layers, 
                 attn_heads, 
                 dropout
                ):
        """
        :param vocab_size: vocab_size of total words
        :param max_len: maximum sequence length(=512 from paper)
        :param n_segs: number of segments(=2 from paper)
        :param embed_dim: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embeddings = DistilBERTEmbedding(vocab_size, max_len, embed_dim, dropout).cuda()
        
        # embedding for BERT, sum of positional, segment, token embeddings
        self.encoder_blocks = nn.ModuleList([
            EncoderLayer(vocab_size, max_len, embed_dim, attn_heads, n_layers, dropout).to(torch.device("cuda")) 
            for _ in range(n_layers//2)
        ]).cuda()
    
    def forward(self, x):
        # attention masking for padded token
        # (batch_size, seq_len, seq_len) is accepted by nn.MHA

        x = x.cuda()  # Move input to the same device as embeddings

        mask = (x==0).cuda()
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embeddings(x).to(torch.device("cuda"))

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask).to(torch.device("cuda"))
        return x

# Add a classifier on top of the DistilBERT model,  to adapt it for classification tasks.
class DistilBERTClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len,
                 n_segs,
                 embed_dim, 
                 n_layers, 
                 attn_heads, 
                 dropout,
                 num_classes
                ):
        super().__init__()
        self.distilbert = DistilBERT(vocab_size, max_len, n_segs, embed_dim, n_layers, attn_heads, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes).cuda()
        
    def forward(self, x):
        x = self.distilbert(x).to(torch.device("cuda"))
        x = x[:, 0, :].to(torch.device("cuda")) #use the first token of the sequence as the sentence embedding
        x = self.classifier(x).to(torch.device("cuda"))
        return x