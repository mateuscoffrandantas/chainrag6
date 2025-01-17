import os
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

# Adicionar o nome do aplicativo
st.title("Q&A com IA - PLN usando LangChain")

# Componentes interativos
file_input = st.file_uploader("Upload a file", type=['txt'])
openaikey = st.text_input("Enter your OpenAI API Key", type='password')
prompt = st.text_area("Enter your questions", height=160)
run_button = st.button("Run!")

select_k = st.slider("Number of relevant chunks", min_value=1, max_value=5, value=2)
select_chain_type = st.radio("Chain type", ['stuff', 'map_reduce', "refine", "map_rerank"])

# Função para carregar documentos
def load_document(file_path):
    loader = TextLoader(file_path)
    return loader.load()

# Função de perguntas e respostas
def qa(file_path, query, chain_type, k):
    try:
        documents = load_document(file_path)
        
        # split the documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        texts = text_splitter.split_documents(documents)
        
        # select which embeddings we want to use
        embeddings = OpenAIEmbeddings()
        
        # create the vectorestore to use as the index
        db = DocArrayInMemorySearch.from_documents(texts, embeddings)
        
        # expose this index in a retriever interface
        retriever = db.as_retriever(search_kwargs={"k": k})
        
        # create a chain to answer questions 
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type=chain_type, 
            retriever=retriever, 
            return_source_documents=True
        )
        result = qa({"query": query})
        return result
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Função para exibir o resultado no Streamlit
def display_result(result):
    if result:
        st.markdown("### Result:")
        st.write(result["result"])
        st.markdown("### Relevant source text:")
        for doc in result["source_documents"]:
            st.markdown("---")
            st.markdown(doc.page_content)

# Execução do app
if run_button and file_input and openaikey and prompt:
    with st.spinner("Running QA..."):
        # Salvar o arquivo em um local temporário
        temp_file_path = os.path.join(tempfile.gettempdir(), file_input.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_input.read())

        # Configurar a chave de API do OpenAI
        os.environ["OPENAI_API_KEY"] = openaikey

        try:
            # Executar a função de perguntas e respostas
            result = qa(temp_file_path, prompt, select_chain_type, select_k)
            # Exibir o resultado
            display_result(result)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
