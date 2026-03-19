import torch
import torch.nn as nn

class ConceptProjector(nn.Module):
    def __init__(self, concept_memory_path="concept_memory_v1.pt", llm_d_model=512):
        super().__init__()
        # 1. 載入預先計算好的記憶矩陣
        memory_data = torch.load(concept_memory_path)
        # memory_matrix 形狀: [Num_Concepts, Memory_Dim(通常是384)]
        self.register_buffer("memory_matrix", memory_data["memory_matrix"])
        
        memory_dim = memory_data["dim"]
        
        # 2. 投影層：將記憶向量轉換為與你的 LLM 相同的維度 (例如 384 -> 512)
        self.projector = nn.Sequential(
            nn.Linear(memory_dim, llm_d_model),
            nn.GELU(),
            nn.Linear(llm_d_model, llm_d_model),
            nn.LayerNorm(llm_d_model)
        )
        
    def forward(self, concept_probs):
        """
        concept_probs: [Batch, Num_Concepts] -> 來自 vision_bridge
        """
        # 3. 核心邏輯：加權求和 (Weighted Sum)
        # 根據機率，把所有激活的概念記憶「揉」成一個 Context Vector
        # [Batch, Num_Concepts] @ [Num_Concepts, Memory_Dim] -> [Batch, Memory_Dim]
        weighted_memory = torch.matmul(concept_probs, self.memory_matrix)
        
        # 4. 投影到 LLM 空間
        # [Batch, llm_d_model]
        visual_context = self.projector(weighted_memory)
        
        return visual_context

# --- 整合到 LLM 的範例 ---
class CML_Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 假設這是你原本的 Attention
        self.attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        # ... 其他層 ...

    def forward(self, x, visual_context):
        """
        x: 文字 Token [Batch, Seq_Len, d_model]
        visual_context: 視覺投影 [Batch, d_model]
        """
        # 將視覺上下文「注入」到文字序列中
        # 這裡有兩種做法：
        # A. 直接相加 (像 Position Embedding)
        # B. 作為 Attention 的額外參考 (更強大)
        
        # 簡單做法：將 visual_context 擴展維度並與文字相加
        x = x + visual_context.unsqueeze(1) 
        
        # 後續接 Attention
        attn_out, _ = self.attn(x, x, x)
        return x + attn_out