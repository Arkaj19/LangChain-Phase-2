from langchain_community.document_loaders import WebBaseLoader

url = 'https://leetcode.com/problems/two-sum/description/'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print( docs[0])