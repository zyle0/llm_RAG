from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding import ZhipuAIEmbeddings
from llm import ZhipuAILLM
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import argparse

def loadDocuments():
    loader = TextLoader(args.doc_path)
    pages = loader.load()
    return pages

def getLLM():
    llm = ZhipuAILLM(model="chatglm_turbo", zhipuai_api_key=args.api_key, temperature=0)
    return llm

def getEmbedding():
    embedding = ZhipuAIEmbeddings(zhipuai_api_key=args.api_key)
    return embedding

def docSplit(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.CHUNK_SIZE,
        chunk_overlap=args.OVERLAP_SIZE
    )
    split_docs = text_splitter.split_documents(pages)
    return split_docs

def vectorization(split_docs, embedding):
    print(f"正在处理文档...")
    vectordb = Chroma.from_documents(
        documents=split_docs, 
        embedding=embedding,
        persist_directory=args.persist_directory
    )
    print(f"文档处理完成!")
    return vectordb

def getPrompt():
    # Build prompt
    template = """仅使用以下上下文片段来回答最后的问题。如果你不知道答案，只需说不知道，尽量简明扼要地回答。
    上下文片段: {context}
    问题：{question}"""
    prompt = PromptTemplate.from_template(template)
    return prompt

def initChain(llm, vectordb, prompt):
    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def run():
    # 加载文档
    pages = loadDocuments()
    # 加载大语言模型
    llm = getLLM()
    # 加载embedding
    embedding = getEmbedding()
    # 文本分割
    split_docs = docSplit(pages)
    # 加载向量数据库并文本向量化
    vectordb = vectorization(split_docs, embedding)
    # 设计prompt
    prompt = getPrompt()
    # 初始化问答链
    qa_chain = initChain(llm, vectordb, prompt)

    while True:
        question = input("请输入您的问题:")
        result = qa_chain({"query": question})
        print(f"大语言模型的回答：{result['result']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default='',  help = 'chatglm api key')
    parser.add_argument('--doc_path', default='zfgzbg.txt', help = 'document path')
    parser.add_argument('--CHUNK_SIZE', default=500, help = 'Maximum size of chunks to return')
    parser.add_argument('--OVERLAP_SIZE', default=50, help = 'Overlap in characters between chunks')
    parser.add_argument('--persist_directory', default='./data_base/chroma', help = 'vectorstores persist directory')
    args = parser.parse_args()
    run()
