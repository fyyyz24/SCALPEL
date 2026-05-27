import torch
from torch import nn
from llm2vec import LLM2Vec

class LLM2VecTextTransformer(nn.Module):
    def __init__(self, text_proj=None,
                 model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                 peft_path=None,
                 enable_bidirectional=True,
                 torch_dtype=torch.bfloat16):
        super().__init__()
        # If peft_path is empty string, treat as None (no PEFT weights)
        if peft_path is not None and peft_path.strip() == "":
            peft_path = None
        self.text = LLM2Vec.from_pretrained(
                model_name_or_path,
                peft_path,
                merge_peft = True if peft_path else False,
                enable_bidirectional=enable_bidirectional,
                attn_implementation = "flash_attention_2",
                torch_dtype=torch_dtype
            )
        self.text_proj = text_proj
        
    def lock(self, **kwargs):
        for param in self.text.parameters():
            param.requires_grad = False
            
    def forward(self, text, batch_size=32): 
        with torch.autocast("cuda"):        
            x = self.text.encode(text,batch_size=batch_size).to(torch.float16)
            if self.text_proj is not None:
                x = self.text_proj(x, l2_norm=False)
        return x
    
    def set_grad_checkpointing(self, enable=True):
        #Not implemented
        pass 