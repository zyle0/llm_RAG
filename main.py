from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding import ZhipuAIEmbeddings
from llm import ZhipuAILLM
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


api_key = ''
doc_path = "zfgzbg.txt"
CHUNK_SIZE = 500
OVERLAP_SIZE = 50
persist_directory = './data_base/chroma'

embedding = ZhipuAIEmbeddings(zhipuai_api_key=api_key)
llm = ZhipuAILLM(model="chatglm_turbo", zhipuai_api_key=api_key,temperature=0)

loader = TextLoader(doc_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
split_docs = text_splitter.split_documents(pages)
print(f"切分后的文件数量：{len(split_docs)}")
print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
print(f"正在处理文档...")
vectordb = Chroma.from_documents(
    documents=split_docs, 
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
print(f"文档处理完成!")

# Build prompt
template = """使用以下上下文片段来回答最后的问题。如果你不知道答案，只需说不知道，不要试图编造答案。答案最多使用三个句子。尽量简明扼要地回答。
上下文片段: {context}.
问题：{question}.
有用的回答："""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

while True:
    question = input("请输入您的问题:")
    result = qa_chain({"query": question})
    print(f"大语言模型的回答：{result['result']}")
