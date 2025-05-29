import pandas as pd
import re
import nltk
import spacy
import logging
from nltk.corpus import stopwords

# 🔹 1. 确保 NLTK 资源已下载
nltk.download('stopwords')
nltk.download('punkt')

# 🔹 2. 配置文件路径
input_path = "/Users/chenxiaoting/Downloads/我的文件/博士学习/大论文/论文数据/txt 2025/china.csv"
output_tokens_path = "/Users/chenxiaoting/Downloads/我的文件/博士学习/大论文/论文数据/txt 2025/Pre/china_cleaned_tokens_lemm.csv"
output_meta_path = "/Users/chenxiaoting/Downloads/我的文件/博士学习/大论文/论文数据/txt 2025/Pre/china_metadata.csv"
extra_stopwords_path = "/Users/chenxiaoting/Downloads/我的文件/博士学习/大论文/论文数据/txt 2025/stopwords-en.txt"

# 🔹 3. 自定义停用词（**所有的词会进行词性还原**）
raw_custom_stopwords = set([
    'china', 'chinese', 'usa', 'america', 'russia', 'country', 'year',
    '2020', '2021', '2022', '2023', '2024', 'people', 'government',
    'population', 'percent', 'new', 'said', 'would', 'could', 'also', 'many',
    'including', 'however', 'say', 'make', 'number', 'need', 'high', 'item',
    'subject', 're', 'edu', 'use', 'million', 'billion', 'yuan','due',
    'according', 'get', 'right', 'leave', 'should', 'data', 'based','day','photo'
])

# 加载 SpaCy 进行词性还原（lemmatization）
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# 🔹 4. 读取 `stopwords-en.txt` 并进行词性还原
try:
    with open(extra_stopwords_path, 'r', encoding='utf-8') as file:
        file_stopwords = set(line.strip() for line in file if line.strip())  # 去除空格和空行
        print("✅ Stopwords file successfully loaded.")
except FileNotFoundError:
    file_stopwords = set()
    print("⚠️ Stopwords file not found! Using only NLTK and custom stopwords.")

# 🔹 5. 对所有停用词进行词性还原（Lemmatization）
custom_stopwords = set(token.lemma_ for word in raw_custom_stopwords for token in nlp(word))
lemmatized_file_stopwords = set(token.lemma_ for word in file_stopwords for token in nlp(word))

# 🔹 6. 结合 NLTK、文件、自定义停用词（最终的 stopwords 集合）
stop_words = set(stopwords.words('english')) | custom_stopwords | lemmatized_file_stopwords

print(f"🔹️ 停用词总数：{len(stop_words)}")
print(f"🔹️ 示例停用词（前20个）：{list(stop_words)[:20]}")

# 🔹 7. 文本清洗和词性还原
def clean_and_lemmatize(text):
    if not isinstance(text, str):
        return ''  # 不是字符串，返回空值

    # 清理文本中的 URL、HTML 标签、邮箱等
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # 只保留英文字母
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = text.lower()

    # 词形还原 + 停用词过滤
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.lemma_ not in stop_words and token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']]

    # 调试输出
    print(f"🔹️ 原始文本: {text[:100]}")  # 输出前100个字符
    print(f"🔹️ 清洗后文本: {' '.join(words)[:100]}")  # 输出处理后前100个字符

    return ' '.join(words)

# 🔹 8. 日志记录
logging.basicConfig(filename='preprocess_en.log', level=logging.INFO)
logging.info("Processing started...")

# 🔹 9. 处理英语 CSV 文件
def process_english_csv(input_path):
    print(f"🔹 开始处理文件：{input_path}")
    df = pd.read_csv(input_path)

    # 确保 CSV 包含需要的列
    required_columns = ['source', 'title', 'date', 'content']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''

    # 合并标题和文本内容
    print("🔹 正在合并标题和文本...")
    df['full_content'] = df['title'].astype(str) + ' ' + df['content'].astype(str)

    # 进行文本清洗和词性还原
    print("🔹 正在清洗和词性还原...")
    df['cleaned_text'] = df['full_content'].apply(clean_and_lemmatize)

    # 生成 LDA 数据（分词文本）
    lda_data = df[['cleaned_text']].rename(columns={'cleaned_text': 'tokens'})
    lda_data['doc_id'] = lda_data.index

    # 生成元数据
    meta_data = df[['source', 'date']].copy()
    meta_data['doc_id'] = meta_data.index

    # 保存处理后的 CSV 文件
    lda_data.to_csv(output_tokens_path, index=False, encoding='utf-8-sig')  # UTF-8-SIG 解决 Excel 乱码问题
    meta_data.to_csv(output_meta_path, index=False, encoding='utf-8-sig')

    logging.info(f"Processing completed! Tokens saved to {output_tokens_path}")
    logging.info(f"Metadata saved to {output_meta_path}")
    print(f"🎉 清洗完成！分词文本已保存到：{output_tokens_path}")
    print(f"👌 元数据已保存到：{output_meta_path}")

if __name__ == "__main__":
    process_english_csv(input_path)
