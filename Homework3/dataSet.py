import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizerSrc, tokenizerTarget, srcLang, targetLang, seqLen):
        super().__init__()
        self.seq_len = seqLen

        self.ds = ds
        self.tokenizer_src = tokenizerSrc
        self.tokenizer_tgt = tokenizerTarget
        self.src_lang = srcLang
        self.tgt_lang = targetLang

        self.sos_token = torch.tensor([tokenizerTarget.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizerTarget.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizerTarget.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        srcTargetPair = self.ds[idx]
        srcText = srcTargetPair['translation'][self.src_lang]
        targetText = srcTargetPair['translation'][self.tgt_lang]
        encInTokens = self.tokenizer_src.encode(srcText).ids
        decInTokens = self.tokenizer_tgt.encode(targetText).ids
        encNumPad = self.seq_len - len(encInTokens) - 2  
        decNumPad = self.seq_len - len(decInTokens) - 1
        if encNumPad < 0 or decNumPad < 0:
            raise ValueError("Sentence is too long")

        encoderIn = torch.cat(
            [
                self.sos_token,
                torch.tensor(encInTokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encNumPad, dtype=torch.int64),
            ],
            dim=0,
        )
        assert encoderIn.size(0) == self.seq_len
        decoderIn = torch.cat(
            [
                self.sos_token,
                torch.tensor(decInTokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decNumPad, dtype=torch.int64),
            ],
            dim=0,
        )
        assert decoderIn.size(0) == self.seq_len
        label = torch.cat(
            [
                torch.tensor(decInTokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decNumPad, dtype=torch.int64),
            ],
            dim=0,
        )
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoderIn, "decoder_input": decoderIn, 
            "encoder_mask": (encoderIn != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoderIn != self.pad_token).unsqueeze(0).int() & causal_mask(decoderIn.size(0)), 
            "label": label, "src_text": srcText,"target_text": targetText,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0