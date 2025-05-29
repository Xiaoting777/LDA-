import pandas as pd
import re
import nltk
import logging
import pymorphy2  # 处理俄语词形还原
from nltk.corpus import stopwords

# 🔹 1. 确保 NLTK 资源已下载
nltk.download('stopwords')
nltk.download('punkt')

# 🔹 2. 配置文件路径
input_path = "/Users/chenxiaoting/Downloads/我的文件/博士学习/大论文/论文数据/txt 2025/RU.csv"
output_tokens_path = "/Users/chenxiaoting/Downloads/我的文件/博士学习/大论文/论文数据/txt 2025/RU_cleaned_tokens_lemm.csv"
output_meta_path = "/Users/chenxiaoting/Downloads/我的文件/博士学习/大论文/论文数据/txt 2025/RU_metadata.csv"
stopwords_russian_path = "/Users/chenxiaoting/Downloads/我的文件/博士学习/大论文/论文数据/txt 2025/stopwords_russian.txt"

# 🔹 3. 自定义停用词（**所有的词会进行词形还原**）
raw_custom_stopwords = set([
    'млн','тысяча','млрд','россия', 'кнр','китай', 'страна', 'год', '2020', '2021', '2022', '2023', '2024',
    'люди', 'правительство', 'население','сайт','являться','месяц','поэтому','фото','декабрь',
    'процент','новый', 'сказать','говориться','Китайцев','говорить', 'могу', 'еще', 'включая',
    'однако', 'политика', 'демография','тасс','риа', 'хоть', 'между', 'впрочем', 'этот', 'весь',
    'такой', 'который', 'свой', 'наш', 'ваш', 'уже', 'ещё', 'либо','сделать','давать','дать',
    'январь','июнь','сообщать','сообщить','ссылка'
])

# 初始化词形还原器
morph = pymorphy2.MorphAnalyzer()

# 🔹 4. 读取 `stopwords_russian.txt` 并进行词形还原
try:
    with open(stopwords_russian_path, 'r', encoding='utf-8') as file:
        file_stopwords = set(line.strip() for line in file if line.strip())  # 去除空格和空行
        print("✅ Russian stopwords file successfully loaded.")
except FileNotFoundError:
    file_stopwords = set()
    print("⚠️ Stopwords file not found! Using only NLTK and custom stopwords.")

# 🔹 5. 对所有停用词进行词性还原（Lemmatization）
custom_stopwords = set(morph.parse(word)[0].normal_form for word in raw_custom_stopwords)
lemmatized_file_stopwords = set(morph.parse(word)[0].normal_form for word in file_stopwords)

# 🔹 6. 结合 NLTK、文件、自定义停用词（最终的 stopwords 集合）
stop_words = set(stopwords.words('russian')) | custom_stopwords | lemmatized_file_stopwords
print(f"🔹 总共加载 {len(stop_words)} 个停用词。")

# 🔹 7. 额外规则：部分前缀匹配的停用词
def is_stopword(word):
    stopword_roots = ['китай', 'росси']  # 过滤掉所有以这些前缀开头的词
    return any(word.startswith(root) for root in stopword_roots)

# 🔹 8. 文本清洗与词性还原
def clean_and_lemmatize(text):
    if not isinstance(text, str):
        return ''  # 不是字符串，返回空值

    # 清理文本中的 URL、HTML 标签、邮箱等
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\u0400-\u04FF\s]', ' ', text)  # 只保留俄文字母
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = text.lower()

    # 词形还原 + 停用词过滤
    words = [
        morph.parse(word)[0].normal_form
        for word in text.split()
        if morph.parse(word)[0].normal_form not in stop_words and not is_stopword(word)
    ]

    return ' '.join(words)

# 🔹 9. 日志记录
logging.basicConfig(filename='preprocess_ru.log', level=logging.INFO)
logging.info("Processing started...")

# 🔹 10. 处理俄文 CSV 文件
def process_russian_csv(input_path):
    print(f"🔹 开始处理文件：{input_path}")
    df = pd.read_csv(input_path, encoding='utf-8')  # 指定编码 UTF-8

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
    process_russian_csv(input_path)
