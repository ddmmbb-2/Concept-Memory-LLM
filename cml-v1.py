import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# 假設 PoC 規格
CML_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 12,
    "vocab_size": 8192,
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
        # x: [B, L, D], visual_context: [B, D]
        
        # 1. 視覺注入：將視覺 Context 透過門控融入 Token
        # 我們將視覺向量擴展到每一個 Token 上
        v_gate = self.visual_gate(x) 
        x = x + (visual_context.unsqueeze(1) * v_gate)
        
        # 2. 標準 Transformer 層
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        
        x = x + self.mlp(self.ln2(x))
        return x

class ConceptMemoryLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        
        # 文字 Embedding
        self.token_embedding = nn.Embedding(config["vocab_size"], self.d_model)
        
        # 🌟 核心組件：ConceptProjector (將 384維記憶 轉為 512維語義)
        # 這裡直接內建，簡化整合
        memory_data = torch.load("concept_memory_v1.pt")
        self.register_buffer("concept_matrix", memory_data["memory_matrix"]) # [Concepts, 384]
        self.projector = nn.Linear(config["memory_dim"], self.d_model)
        
        # 堆疊 Block
        self.blocks = nn.ModuleList([
            CML_Block(self.d_model, config["n_heads"]) for _ in range(config["n_layers"])
        ])
        
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, config["vocab_size"], bias=False)
        
        # ✨ Weight Tying (3060 救星)
        self.head.weight = self.token_embedding.weight
        
    def forward(self, idx, concept_probs):
        B, L = idx.shape
        
        # 1. 處理視覺記憶
        # weighted_memory = [B, 384]
        weighted_memory = torch.matmul(concept_probs, self.concept_matrix)
        # visual_context = [B, 512]
        visual_context = self.projector(weighted_memory)
        
        # 2. 文字處理
        x = self.token_embedding(idx) # [B, L, 512]
        
        # 3. 逐層推理 (支援 Checkpoint 以省顯存)
        for block in self.blocks:
            if self.training:
                x = checkpoint(block, x, visual_context, use_reentrant=False)
            else:
                x = block(x, visual_context)
                
        logits = self.head(self.ln_f(x))
        return logits