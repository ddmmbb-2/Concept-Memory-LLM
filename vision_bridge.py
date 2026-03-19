import torch
import clip
from PIL import Image
import torch.nn.functional as F

class VisionBridge:
    def __init__(self, concept_memory_path="concept_memory_v1.pt", device="cuda"):
        self.device = device
        print(f"👁️ 正在載入視覺編碼器 (CLIP)...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # 載入先前生成的記憶資料
        print(f"📂 載入概念清單: {concept_memory_path}")
        memory_data = torch.load(concept_memory_path)
        self.concept_list = memory_data["concept_list"]
        
        # 預先計算概念的文本 Embedding (Zero-shot Classifier)
        self.concept_features = self._precompute_concepts()

    def _precompute_concepts(self):
        """將 'dog' 變成 'a photo of a dog' 並計算其 CLIP 特徵"""
        print(f"🧪 正在預計算 {len(self.concept_list)} 個概念的視覺特徵...")
        # 使用 Prompt Engineering 增加準確度
        templates = [f"a photo of a {c}" for c in self.concept_list]
        text_tokens = clip.tokenize(templates).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # 標準化 (L2 Normalize) 方便之後算 Cosine Similarity
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def get_concept_activation(self, image_path, temperature=0.01, top_k=5):
        """輸入圖片路徑，輸出概念激活機率向量"""
        # 1. 圖片預處理
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 2. 提取圖片特徵
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 3. 計算相似度 (Dot Product on normalized vectors = Cosine Similarity)
            # logits: [1, Num_Concepts]
            logits = (image_features @ self.concept_features.T) / temperature
            
            # 4. 轉化為機率分布
            probs = F.softmax(logits, dim=-1).squeeze(0)
            
        # 取得最強的 Top-K 概念方便 Debug
        top_probs, top_idxs = probs.topk(top_k)
        activated_concepts = {self.concept_list[idx]: prob.item() for prob, idx in zip(top_probs, top_idxs)}
        
        return probs, activated_concepts

# --- 測試代碼 ---
if __name__ == "__main__":
    # 假設你有一張名為 test.jpg 的圖片
    bridge = VisionBridge(device="cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        probs, debug_info = bridge.get_concept_activation("test.jpg")
        print("\n🔥 激發概念排行榜:")
        for concept, score in debug_info.items():
            print(f"[{concept}]: {score:.4f}")
            
        # 這個 probs 就是之後要餵給 LLM 的 Tensor [Num_Concepts]
        print(f"\n✅ 激活向量維度: {probs.shape}")
    except FileNotFoundError:
        print("❌ 請準備一張 test.jpg 進行測試。")