import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from vision_bridge import VisionBridge
from model_integration import ConceptMemoryLLM, CML_CONFIG
from PIL import Image
import os

# 配置
cfg = {
    "model_path": "cml_v1_final.pth",
    "tokenizer_path": "bpe_tokenizer_v12.json",
    "concept_memory": "concept_memory_v1.pt",
    "device": "cuda",
    "max_new_tokens": 30,
    "temperature": 0.7
}

class CMLInference:
    def __init__(self):
        self.device = cfg["device"]
        self.tokenizer = Tokenizer.from_file(cfg["tokenizer_path"])
        
        # 1. 載入視覺橋樑
        self.bridge = VisionBridge(cfg["concept_memory"], device=self.device)
        
        # 2. 載入模型
        self.model = ConceptMemoryLLM(CML_CONFIG).to(self.device)
        if os.path.exists(cfg["model_path"]):
            self.model.load_state_dict(torch.load(cfg["model_path"], weights_only=True, map_location=self.device))
            print(f"✅ 已成功載入權重: {cfg['model_path']}")
        else:
            print(f"⚠️ 找不到權重檔案 {cfg['model_path']}，將使用隨機初始化模型（僅供測試結構用）")
        
        self.model.eval()

    def generate(self, image_path, prompt=""):
        # A. 視覺感知
        probs, activated = self.bridge.get_concept_activation(image_path)
        probs = probs.unsqueeze(0).to(self.device)
        
        print(f"👁️ 模型看見了概念: {list(activated.keys())[:5]}")

        # B. 準備輸入 Token
        tokens = self.tokenizer.encode(prompt).ids
        if len(tokens) == 0:
            bos_id = self.tokenizer.token_to_id("<|endoftext|>")
            tokens = [bos_id] if bos_id is not None else [0]

        x = torch.tensor([tokens], dtype=torch.long).to(self.device)

        # C. 自回歸生成 (含 Debug 訊息)
        print(f"🚀 開始生成文字 (Temp: {cfg['temperature']})...")
        generated_ids = []

        for _ in range(cfg["max_new_tokens"]):
            with torch.no_grad():
                logits = self.model(x, probs)
                next_token_logits = logits[:, -1, :] / cfg["temperature"]
                
                # 採樣
                soft_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(soft_probs, num_samples=1)
                
                token_id = next_token.item()
                generated_ids.append(token_id)
                
                x = torch.cat((x, next_token), dim=1)
                
                # 結束判定
                if token_id == self.tokenizer.token_to_id("<|endoftext|>"):
                    break
        
        print(f"🔢 生成的 Token IDs: {generated_ids}")
        return self.tokenizer.decode(x[0].tolist())

if __name__ == "__main__":
    test_img = "test_image.jpg"
    if os.path.exists(test_img):
        infer = CMLInference()
        
        # 🌟 關鍵：給它一個開頭，引導它進入「描述模式」
        result = infer.generate(test_img, prompt="A photo of") 
        print(f"\n📝 最終模型描述: {result}")
    else:
        print(f"❌ 找不到圖片")