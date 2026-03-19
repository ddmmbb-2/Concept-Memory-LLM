import json
import os

# 配置路徑
config = {
    "coco_json": "data/coco/annotations/captions_val2017.json",
    "image_dir": "data/coco/val2017",
    "output_json": "data/train_data.json"
}

def extract_coco_data():
    print(f"📖 正在讀取 COCO 原始標籤: {config['coco_json']}...")
    
    with open(config['coco_json'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 建立 Image ID 到 File Name 的映射表
    # COCO 內部是用 ID 溝通，但我們需要檔案名稱
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    print(f"🖼️ 找到 {len(id_to_filename)} 張圖片索引。")

    # 2. 提取描述並與圖片路徑配對
    cml_data = []
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id in id_to_filename:
            file_name = id_to_filename[img_id]
            # 檢查圖片檔案是否真的存在，避免之後訓練報錯
            full_path = os.path.join(config['image_dir'], file_name)
            
            if os.path.exists(full_path):
                cml_data.append({
                    "image": file_name, # 只存檔名，路徑交給 Dataset 處理
                    "caption": ann['caption'].strip()
                })

    print(f"✍️ 提取完成！總共獲得 {len(cml_data)} 組「圖片-文字」配對。")

    # 3. 儲存成我們 CML 專用的簡潔格式
    with open(config['output_json'], 'w', encoding='utf-8') as f:
        json.dump(cml_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 檔案已存至: {config['output_json']}")
    print(f"💡 接下來你可以執行 preprocess_vision.py 來生成視覺特徵了！")

if __name__ == "__main__":
    if os.path.exists(config['coco_json']):
        extract_coco_data()
    else:
        print(f"❌ 找不到標籤檔，請確認路徑: {config['coco_json']}")