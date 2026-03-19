import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 配置
config = {
    "device": "cpu", # 產生記憶矩陣用 CPU 即可，留著 GPU 給以後訓練
    "model_name": "all-MiniLM-L6-v2", # 超輕量但強大的語義模型
    "output_file": "concept_memory_v1.pt",
    "json_debug_file": "concept_memory_debug.json", # 方便人眼查看
    "vector_dim": 384 # MiniLM 的輸出維度，之後 LLM 的 d_model 也要對齊或投影
}

# 1. 定義初始概念清單 (PoC 先做 20 個範例，你可以擴充到 512 個)
# 格式: "概念": "深度描述/常識聯想"
raw_concepts = {
    # 實體 (Entities)
    "dog": "A loyal four-legged pet that barks, loves playing, and guards the home.",
    "cat": "A small independent feline pet that purrs, hunts mice, and likes climbing.",
    "tree": "A tall perennial plant with a wooden trunk, branches, and green leaves providing shade.",
    "car": "A four-wheeled motor vehicle used for transportation on roads.",
    "water": "A transparent, tasteless liquid essential for all forms of life and hydration.",
    
    # 動作 (Actions)
    "running": "Moving rapidly on foot, an energetic physical activity for exercise or speed.",
    "eating": "The act of consuming food to gain energy and nutrition.",
    "broken": "Something damaged into pieces, no longer functioning or in its original state.",
    
    # 屬性/狀態 (Attributes/States)
    "danger": "A situation or object that can cause harm, injury, or requires high alertness.",
    "happy": "An emotional state of joy, satisfaction, and positive well-being.",
    "cold": "A low temperature state, often associated with ice, snow, or shivering.",
    "old": "Having lived or existed for a long time, showing signs of wear or history.",
    
    # 場景 (Scenes)
    "indoor": "Inside a building or room, protected from outside weather.",
    "forest": "A large area covered chiefly with trees and undergrowth, a natural habitat.",
    "night": "The period from sunset to sunrise when it is dark outside."
}

def generate_memory_bank():
    print(f"🚀 正在載入語義編碼器: {config['model_name']}...")
    encoder = SentenceTransformer(config['model_name']).to(config['device'])
    
    concepts = list(raw_concepts.keys())
    descriptions = list(raw_concepts.values())
    
    print(f"🧠 正在將 {len(concepts)} 個概念轉化為記憶向量...")
    # 將描述文字轉化為向量
    with torch.no_grad():
        memory_embeddings = encoder.encode(descriptions, convert_to_tensor=True)
    
    # 2. 建立記憶字典 (供人眼檢查)
    memory_dict = {}
    for i, concept in enumerate(concepts):
        memory_dict[concept] = {
            "id": i,
            "description": descriptions[i],
            "vector_sample": memory_embeddings[i][:5].tolist() # 只存前5維觀察
        }
    
    # 3. 儲存為 PyTorch 格式 (供模型直接讀取)
    payload = {
        "concept_list": concepts,
        "memory_matrix": memory_embeddings.cpu(), # [N, 384]
        "dim": config['vector_dim']
    }
    
    torch.save(payload, config['output_file'])
    
    with open(config['json_debug_file'], 'w', encoding='utf-8') as f:
        json.dump(memory_dict, f, indent=4, ensure_update=False)

    print(f"✅ 成功！記憶矩陣已存至 {config['output_file']}")
    print(f"📊 矩陣形狀: {memory_embeddings.shape} (概念數 x 向量維度)")

if __name__ == "__main__":
    generate_memory_bank()