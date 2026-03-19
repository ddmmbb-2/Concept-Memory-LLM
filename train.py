import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model_integration import ConceptMemoryLLM, CML_CONFIG # 引用之前寫的模型架構
from tqdm import tqdm
import json
import os

# 訓練配置
train_cfg = {
    "lr": 1e-4,
    "batch_size": 16, # 3060 關閉 CLIP 後可以開比較大
    "epochs": 10,
    "max_length": 64,
    "vision_feat_dir": "data/vision_feats",
    "train_json": "data/train_data.json",
    "tokenizer_path": "bpe_tokenizer_v12.json"
}

class CMLDataset(Dataset):
    def __init__(self, json_path, vision_dir, tokenizer_path, max_len):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.vision_dir = vision_dir
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=max_len)
        self.tokenizer.enable_truncation(max_length=max_len)
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 1. 載入文字 Token
        tokens = self.tokenizer.encode(item['caption']).ids
        
        # 2. 載入預處理好的視覺向量
        feat_path = os.path.join(self.vision_dir, item['image'] + ".pt")
        vision_probs = torch.load(feat_path) # [512]
        
        return torch.tensor(tokens), vision_probs

def train():
    device = "cuda"
    # 1. 準備數據
    ds = CMLDataset(train_cfg["train_json"], train_cfg["vision_feat_dir"], 
                    train_cfg["tokenizer_path"], train_cfg["max_length"])
    dl = DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=True)

    # 2. 初始化模型
    model = ConceptMemoryLLM(CML_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    
    print(f"🚀 CML V1 啟動訓練！參數規模: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    model.train()
    for epoch in range(train_cfg["epochs"]):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}")
        for x, v_probs in pbar:
            x, v_probs = x.to(device), v_probs.to(device)
            
            # Label 是 x 右移一位 (Causal Language Modeling)
            logits = model(x[:, :-1], v_probs)
            targets = x[:, 1:]
            
            loss = F.cross_entropy(logits.reshape(-1, CML_CONFIG["vocab_size"]), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    torch.save(model.state_dict(), "cml_v1_final.pth")
    print("🎉 訓練完成！模型已儲存。")

if __name__ == "__main__":
    train()