import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

# 🚀 CML-V1 核心配置
CML_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 12,
    "vocab_size": 16384, # 請確保與你的 BPE Tokenizer 詞表大小一致
    "concept_num": 512,  # 你的概念清單長度
    "memory_dim": 384,   # MiniLM 的輸出維度
}

class CML_Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        
        # 🌊 語義門控：控制視覺記憶對這一層的影響程度
        self.visual_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x, visual_context):
        # 1. 視覺注入
        v_gate = self.visual_gate(x) 
        x = x + (visual_context.unsqueeze(1) * v_gate)
        
        # 2. Attention
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        
        # 3. MLP
        x = x + self.mlp(self.ln2(x))
        return x

class ConceptMemoryLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        
        self.token_embedding = nn.Embedding(config["vocab_size"], self.d_model)
        
        # 載入預先計算好的記憶矩陣
        # 確保 concept_memory_v1.pt 在同一個目錄下
        memory_data = torch.load("concept_memory_v1.pt", map_location='cpu')
        self.register_buffer("concept_matrix", memory_data["memory_matrix"])
        
        self.projector = nn.Linear(config["memory_dim"], self.d_model)
        
        self.blocks = nn.ModuleList([
            CML_Block(self.d_model, config["n_heads"]) for _ in range(config["n_layers"])
        ])
        
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, config["vocab_size"], bias=False)
        
        # Weight Tying
        self.head.weight = self.token_embedding.weight
        
    def forward(self, idx, concept_probs):
        # 1. 處理視覺記憶
        # 🌟 修正點：確保 concept_probs 與 concept_matrix 的類型一致 (通常轉成 float32)
        concept_probs = concept_probs.to(self.concept_matrix.dtype)
        
        # 現在兩者都是 Float32 了，可以安全乘法
        weighted_memory = torch.matmul(concept_probs, self.concept_matrix)
        
        # 2. 投影到 LLM 空間
        visual_context = self.projector(weighted_memory)
        
        # 3. 文字處理
        x = self.token_embedding(idx)
        
        # 4. 逐層推理
        for block in self.blocks:
            if self.training:
                # 傳入 visual_context
                x = checkpoint(block, x, visual_context, use_reentrant=False)
            else:
                x = block(x, visual_context)
                
        logits = self.head(self.ln_f(x))
        return logits