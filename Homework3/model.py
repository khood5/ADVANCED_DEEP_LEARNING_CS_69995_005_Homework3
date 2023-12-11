import torch 
import torch.nn as nn
import math

# input embedding
class textInputEmbedding(nn.Module):
    # sizeOfEmbedding = d_model in his tutoral 
    def __init__(self, sizeOfEmbedding: int, vocabSize: int) -> None:
        super().__init__()
        self.size_of_embedding = sizeOfEmbedding
        self.vocab_size = vocabSize
        self.embedding = nn.Embedding(self.vocab_size, self.size_of_embedding) # maps words to vectors like a hash map

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.size_of_embedding) #  math.sqrt helps with the convergence 
    
class positionEncoding(nn.Module):
    def __init__(self, sizeOfEmbedding:int, sectanceLength: int, dropout: float, scale=10000.0) -> None:
        super().__init__()
        self.size_of_embedding = sizeOfEmbedding
        self.length_of_sectance = sectanceLength
        self.dropout = nn.Dropout(dropout)
        self.position_encoding_scaling = scale

        # create matrix for each postion encoding (length_of_sectance * size_of_embedding)
        positionEncodingMatrix = torch.zeros(self.length_of_sectance, self.size_of_embedding)
        aPosition = torch.arange(0, self.length_of_sectance, dtype=torch.float32).unsqueeze(1)
        # The value 10000 is chosen arbitrarily, it is a hyperparameter that can be tuned to scale the position encoding
        div = torch.exp(torch.arange(0, self.size_of_embedding, 2).float() * (-math.log(self.position_encoding_scaling) / self.size_of_embedding))
        
        positionEncodingMatrix[:, 0::2] = torch.sin(aPosition*div)
        positionEncodingMatrix[:, 1::2] = torch.cos(aPosition*div)

        self.positionEncodingMatrix = positionEncodingMatrix.unsqueeze(0)

        # self.register_buffer('positionEncodingMatrix', positionEncodingMatrix)
        self.register_buffer('pe', positionEncodingMatrix)
    def forward(self, x):
        x = x + (self.positionEncodingMatrix[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
 
class addNormLayer(nn.Module):
    # 10**-6 is used to just match with tuoral so testing should be the similar 
    def __init__(self, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / (std+self.eps) + self.bias
    
class ffLayer(nn.Module):
    def __init__(self, sizeOfEmbedding: int, sizeFf: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(sizeOfEmbedding, sizeFf)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(sizeFf, sizeOfEmbedding)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class multiHeadAttentionLayer(nn.Module):
    # h = numberOfHead
    def __init__(self, sizeOfEmbedding: int, numberOfHead:int, dropout: float) -> None:
        super().__init__()
        assert sizeOfEmbedding % numberOfHead == 0
        self.size_of_embedding = sizeOfEmbedding
        self.number_of_heads = numberOfHead
        self.dropout = nn.Dropout(dropout)

        self.size_of_individual_q_k_v = self.size_of_embedding // self.number_of_heads # d_k = width of key, query, and value vectors length is the same as the sequance 
        self.wights_q = nn.Linear(self.size_of_embedding, self.size_of_embedding)
        self.wights_k = nn.Linear(self.size_of_embedding, self.size_of_embedding)
        self.wights_v = nn.Linear(self.size_of_embedding, self.size_of_embedding)
        self.wights_output = nn.Linear(self.size_of_embedding,self.size_of_embedding) # concat of heads

        self.attention_scores = None # used in forward

    def forward(self, query, key, value, mask):
        query = self.wights_q(query)
        key = self.wights_k(key)
        value = self.wights_v(value)

        # split into small dim
        query = query.view(query.shape[0], query.shape[1], self.number_of_heads, self.size_of_individual_q_k_v).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.number_of_heads, self.size_of_individual_q_k_v).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.number_of_heads, self.size_of_individual_q_k_v).transpose(1,2)
        
        x, self.attention_scores = multiHeadAttentionLayer.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.sizeOfEmbedding)
        x = self.wights_output(x)
        return x


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # d_k = width of key, query, and value vectors length is the same as the sequance 
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)
        if mask: # apply it
            attention_scores.masked_fill_(mask == 0, -1e9) # -1e9 borrowed from tutroal 
        attention_scores = attention_scores.softmax(dim=-1) 
        
        if dropout: # apply it
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # last value used for viz not needed for model


class residualConnection(nn.Module):
    def __init__(self, dropout:float ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = addNormLayer()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class encoderLayer(nn.Module):
    def __init__(self, selfAttentionBlock: multiHeadAttentionLayer, ffBlock: ffLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = selfAttentionBlock
        self.ff_block = ffBlock
        self.residual_connections = nn.ModuleList([residualConnection(dropout) for _ in range(2)])

    def selfAttention(self, x, src_mask):
        return self.self_attention_block(x, x, x, src_mask)

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, self.selfAttention(x, src_mask))
        x = self.residual_connections[1](x, self.ff_block)
        return x

class encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layer = layers
        self.norm= addNormLayer()

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)



class decoderLayer(nn.Module):
    def __init__(self, selfAttentionBlock: multiHeadAttentionLayer, corssAttentionBlock: multiHeadAttentionLayer, 
                 ffBlock: ffLayer, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = selfAttentionBlock
        self.corss_attention_block = corssAttentionBlock
        self.ff_block = ffBlock

        self.residual_connections = nn.ModuleList([residualConnection(dropout) for _ in range(3)])

    def selfAttention(self, x, encoderOutput, mask):
        return self.self_attention_block(x, encoderOutput, encoderOutput, mask)

    def forward(self, x, encoderOutput, srcMask, targetMask):
        x = self.residual_connections[0](x, self.selfAttention(x, x, targetMask))
        x = self.residual_connections[1](x, self.selfAttention(x, encoderOutput, srcMask))
        x = self.residual_connections[2](x, self.ff_block)

class decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layer = layers
        self.norm= addNormLayer()

    def forward(self, x, encoderOutput, srcMask, targetMask):
        for layer in self.layer:
            x = layer(x, encoderOutput, srcMask, targetMask)
        return self.norm(x)

class projectionLayer(nn.Module):
    def __init__(self, sizeOfEmbedding: int, vocabSize: int) -> None:
        super().__init__()
        self.size_of_embedding = sizeOfEmbedding
        self.vocab_size = vocabSize
        self.projection = nn.Linear(self.size_of_embedding, self.vocab_size)
    
    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)

class transformer(nn.Module):
    def __init__(self, encoder: encoder, decoder: decoder, 
                 srcLangEmbedding: textInputEmbedding, targetLangEmbedding: textInputEmbedding,
                 srcPos: positionEncoding, targetPos: positionEncoding,
                 projLayer: projectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_lang_embedding = srcLangEmbedding
        self.target_lang_embedding = targetLangEmbedding
        self.src_pos = srcPos
        self.target_pos = targetPos
        self.proj_layer = projLayer

    def encode(self, src, src_mask):
        src = self.src_lang_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoderOutput, srcMask, target, targetMask):
        target = self.target_lang_embedding(target)
        target = self.target_pos(target)
        return self.decoder(target, encoderOutput, srcMask, targetMask)
    
    def project(self, x):
        return self.proj_layer(x)

def buildTransformer(srcVocabSize: int, targetVocabSize: int, srcLength:int, targetLength:int,
                     sizeOfEmbedding: int = 512, numberOfBlocks: int = 6, numberOfHeads: int = 8,
                     dropout: float = 0.1, ffSize: int = 2048) -> transformer:
    # make input embedding 
    srcEmbed = textInputEmbedding(sizeOfEmbedding, srcVocabSize)
    targetEmbed = textInputEmbedding(sizeOfEmbedding, targetVocabSize)

    # make pos endocding 
    srcPos = positionEncoding(sizeOfEmbedding, srcLength, dropout)
    targetPos = positionEncoding(sizeOfEmbedding, targetLength, dropout)

    # make n encoder blocks
    encoderBlocks = []
    for _ in range(numberOfBlocks):
        encoderSelfAttentionBlock = multiHeadAttentionLayer(sizeOfEmbedding, numberOfHeads, dropout)
        ffBlock = ffLayer(sizeOfEmbedding, ffSize, dropout)
        encoderBlock = encoderLayer(encoderSelfAttentionBlock, ffBlock, dropout)
        encoderBlocks.append(encoderBlock)

    # make n decoder blocks 
    decoderBlocks = []
    for _ in range(numberOfBlocks):
        decoderSelfAttentionBlock = multiHeadAttentionLayer(sizeOfEmbedding, numberOfHeads, dropout)
        decoderCrossAttentionBlock = multiHeadAttentionLayer(sizeOfEmbedding, numberOfHeads, dropout)
        ffBlock = ffLayer(sizeOfEmbedding, ffSize, dropout)
        decoderBlock = decoderLayer(decoderSelfAttentionBlock, decoderCrossAttentionBlock, ffBlock, dropout)
        decoderBlocks.append(decoderBlock)

    # create en/de coder
    builtEncoder = encoder(nn.ModuleList(encoderBlocks))
    builtDecoder = decoder(nn.ModuleList(decoderBlocks))

    # create projection
    builtProjectionLayer = projectionLayer(sizeOfEmbedding, targetVocabSize)

    # last build the thing
    trans = transformer(builtEncoder, builtDecoder,
                        srcEmbed, targetEmbed,
                        srcPos,targetPos,
                        builtProjectionLayer)
    
    # weight init
    for p in trans.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return trans