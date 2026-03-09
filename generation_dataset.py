# from OpenAI_LLM import OpenAI_LLM
# from kor.extraction import create_extraction_chain
# from kor.nodes import Object, Text, Number
# import tiktoken
import json, time
from tqdm import tqdm
# enc = tiktoken.get_encoding("cl100k_base")
import sys
from pathlib import Path
# 导入 log 模块目录
# sys.path.append("../")
# from log.logutli import Logger

# schema = Object(
#     id="script",
#     description="Adapted from the novel into script",
#     attributes=[
#         Text(
#             id="role",
#             description="The character who is speaking",
#         ),
#         Text(
#             id="dialogue",
#             description="The dialogue spoken by the characters in the sentence",
#         )
#     ],
#     examples=[
#         (
#             '''
#             龙王说∶“再也没有比这更重的兵器了。”悟空不信，和龙王吵了起来，龙婆给龙王说∶“大禹治水时，测定海水深浅的神珍铁最近总是放光，就把这给他，管他能不能用，打发他走算了。”龙王听后告诉悟空∶“这宝物太重了，你自己去取吧！”
#             ''',
#             [
#                 {"role": "龙王", "dialogue": "再也没有比这更重的兵器了。"},
#                 {"role": "龙婆", "dialogue": "大禹治水时，测定海水深浅的神珍铁最近总是放光，就把这给他，管他能不能用，打发他走算了。”龙王听后告诉悟空∶“这宝物太重了，你自己去取吧！"},
#             ],
#         ),
#         (
#             '''
#             悟空见八戒这么长时间不回来，就拔根毫毛变成自己，陪着师父和沙僧，真身驾云来到山凹里，见八戒和妖精正在交战，便高声叫道∶“八戒别慌，老孙来了！”八戒一听，来了精神，没几下，就把那群妖怪打败了。
#             ''',
#             [
#                 {"role": "悟空", "dialogue": "八戒别慌，老孙来了！"},
#             ],
#         )
#     ],
#     many=True,
# )

# 该函数用于读取小说文本，返回字符串形式的文本内容
def read_text(path):
    with open(path, mode='r', encoding='utf-8') as f:
        return f.read()
# 该函数用于将提取的对话数据保存到jsonl文件中，每行一个json对象
def save_data(data):
    filename = path.split('/')[-1].split('.')[0]
    with open(f"./generation_dataset/output/{filename}.jsonl", mode='a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

# 该函数用于将小说文本按token长度分割成多个chunk，返回chunk文本列表
def get_chunk(text):
    """
    text: str
    return: chunk_text
    """
    max_token_len = 600
    chunk_text = []

    curr_len = 0
    curr_chunk = ''

    lines = text.split('\n')  # 假设以换行符分割文本为行
    logger.info('按token分割小说文本')
    for line in lines:
        line_len = len(enc.encode(line))
        if line_len > max_token_len:
            print('warning line_len = ', line_len)
        if curr_len + line_len <= max_token_len:
            curr_chunk += line
            curr_chunk += '\n'
            curr_len += line_len
            curr_len += 1
        else:
            chunk_text.append(curr_chunk)
            curr_chunk = line
            curr_len = line_len
    
    if curr_chunk:
        chunk_text.append(curr_chunk)
    
    return chunk_text

# 该函数用于调用提取链进行对话抽取，并将结果保存到jsonl文件中，包含重试机制
def run(chains, text):
    max_attempts = 3  # 最大尝试次数
    current_attempt = 1
    while current_attempt < max_attempts:
        try:
            response = chains.run(text)
        except Exception as e:
            # print(e)
            logger.error(f"报错：{e}")
        else:
            break
        # finally:
        #     print(f"第 {current_attempt} 次尝试完成。")
        #     current_attempt += 1

    if 'script' in response['data']:
        for item in response['data']['script']:
            # print(item)
            save_data(item)
    else:
        pass

# 该函数用于读取之前保存的jsonl文件中的对话数据，返回对话数据列表,包含多个json对象，每个对象包含角色和对话内容
def read_dialogue(path):
    res = []
    with open(path, mode='r', encoding='utf-8') as file:
        for line in file.readlines():
            res.append(json.loads(line))
    return res

# 该函数用于将提取的对话数据构造成微调数据集的格式，返回微调数据列表，每个元素包含instruction、input和output字段
def save_dataset(path, data):
    output_path = Path(path)
    if output_path.suffix.lower() != '.json':
        output_path = output_path.with_suffix('.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write('\n')

# 该函数用于根据指定的角色列表，从对话数据中提取对应角色的对话内容，并构造成微调数据集的格式，返回微调数据列表
def generate_dataset(data, roles):
    # data:接受完整的对话数据集，列表形式
    # roles:需要提取的对话角色。列表形式
    res = []
    # logger.info('构造微调数据集')
    for i in tqdm(range(1, len(data))):
        role = data[i]['role']
        if role in roles:
            tmp = {
                "instruction": data[i-1]['dialogue'],
                "input": "",
                "output": data[i]['dialogue']
            }
            res.append(tmp)
    return res


if __name__ == "__main__":
    # LOG
    # local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # log_id     = 'generation_dataset'  
    # log_dir    = f'./log/'
    # log_name   = f'generation_dataset_log_{local_time}.log'
    # log_level  = 'info'
    # # 初始化日志
    # logger = Logger(log_id, log_dir, log_name, log_level).logger

    # CONFIG
    path = './dataset/result/zhenhuan01-10.jsonl'  # 小说路径
    # roles = ['克莱恩', '小克']  # 要提取的角色名称
    roles = [ '甄嬛', '钮祜禄·甄嬛','甄玉嬛', '莞贵人', '莞嫔', '甄婉仪', '甄婕妤', '莞贵嫔', '甄昭仪', '莫愁师太', '莞妃', '淑妃', '莞淑妃', '皇贵妃' ]

    # 2、将提取好的对话数据集，改造成微调数据集
    print('======================================抽取完成，构造数据集======================================')
    dialogue_list = read_dialogue(path)
    print(f"共有 {len(dialogue_list)} 条对话样本\n")
    print(f"样本示例 {dialogue_list[0]}")
    # logger.info(f"共有 {len(dialogue_list)} 条对话样本")
    dataset = generate_dataset(dialogue_list, roles)
    print(f"获得 {len(dataset)} 条微调样本")
    # logger.info(f"获得 {len(dataset)} 条微调样本")
    save_dataset(f'dataset/train/lora/{path.split("/")[-1].split(".")[0]}.json', dataset)
    print('======================================构造完成======================================')