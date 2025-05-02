import pickle
import lmdb
import json  # 用于写入 JSON 文件

lmdb_path = "Generated/Chinese/test"
# lmdb_path = "temp/test"

# 打开 LMDB 数据库
lmdb = lmdb.open(
    lmdb_path,
    max_readers=8,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)
print("the lmdb_path is", lmdb_path)

# 创建一个字典来存储所有样本数据
output_data = {}

with lmdb.begin(write=False) as txn:
    # 获取样本数量
    num_sample = 1  # 假设样本数量为 500
    print(f"Number of samples: {num_sample}")
    output_data["num_sample"] = num_sample  # 保存样本数量到字典

    # 遍历所有样本
    samples = {}
    for i in range(num_sample):
        key = str(i).encode("utf-8")  # 假设键是样本的索引
        value = txn.get(key)
        if value:
            try:
                # 尝试反序列化数据
                data = pickle.loads(value)
                print(f"Sample {i}")
                tag_char, coords, fname = (
                    data["tag_char"],
                    data["coordinates"],
                    data["fname"],
                )
                print("the tag_char is", tag_char)
                print("the coords is", coords)
                print("the fname is", fname)
                # 确保数据可序列化为 JSON
                samples[f"Sample_{i}"] = json.loads(json.dumps(data, default=str))
            except Exception as e:
                # 如果反序列化失败，直接保存原始值
                print(f"Sample {i}: Unable to deserialize data, storing raw value.")
                samples[f"Sample_{i}"] = value.decode(
                    "utf-8", errors="ignore"
                )  # 保存原始值为字符串
    output_data["samples"] = samples

# 将数据写入 JSON 文件
output_file = "output.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"All data has been saved to {output_file}")
