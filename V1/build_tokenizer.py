import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def build_tokenizer():
    print("⏳ 正在根據 train_data.json 訓練 BPE 分詞器...")
    
    # 1. 載入資料
    with open("data/train_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['caption'] for item in data]

    # 2. 初始化 BPE
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 3. 設定訓練器
    trainer = trainers.BpeTrainer(
        vocab_size=16384, # 需與 model_integration.py 裡的 CML_CONFIG 一致
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 4. 開始訓練
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # 5. 存檔
    tokenizer.save("bpe_tokenizer_v12.json")
    print("✅ bpe_tokenizer_v12.json 已產生！")

if __name__ == "__main__":
    build_tokenizer()