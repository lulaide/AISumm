import os

# 路径配置 - 使用相对路径
DATA_DIR = "./data/THUCNews"  # 数据文件夹路径
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.txt")  # 训练数据路径
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.txt")  # 测试数据路径

# 模型配置
BERT_MODEL_NAME = "bert-base-chinese"  # Hugging Face模型名称
MODEL_SAVE_DIR = "./save_model"  # 训练时的模型保存路径
SAVED_MODEL_DIR = "./saved_model"  # 最终保存的模型路径
LOG_DIR = "./logs"  # 日志路径

# 数据参数
MAX_LENGTH = 128  # 文本最大长度
TEST_SIZE = 0.2   # 测试集比例
VAL_SIZE = 0.5    # 从测试集中划分验证集的比例（最终train:val:test ≈ 8:1:1）

# 训练参数
TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01