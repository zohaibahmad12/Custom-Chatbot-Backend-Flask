import os
import langchain
from flask import Flask, request, jsonify
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
api_key = os.environ.get('OPEN_AI_API_KEY')
llm = ChatOpenAI(
    api_key=api_key,
    model='gpt-4o-mini',
    temperature=0.9,
    max_tokens=100  # Adjust as needed
)


# urls = [
#    'https://odyssey3d.io/'
# ]
# loader = UnstructuredURLLoader(urls=urls)
# textData=loader.load()



# textSplitter=RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
#  )
# chunks=textSplitter.split_documents(textData)



embeddings=OpenAIEmbeddings(openai_api_key=api_key)
# vectorstore=FAISS.from_documents(chunks,embeddings)

# vectorstore.save_local("vectorstore")
loadedVectorStore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
print(loadedVectorStore)



chain=RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=loadedVectorStore.as_retriever())
# langchain.debug=True
# prompt='who should use odyssey? give answer in max 50 words'
# response=chain.invoke({"question":prompt}, return_only_outputs=True)
# print(response)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    question_with_instruction = f"{question} You can answer in max 80 words."
    print(question_with_instruction)
    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = chain.invoke({"question": question}, return_only_outputs=True)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)