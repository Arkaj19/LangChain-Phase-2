from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = """Cricket is a globally popular bat-and-ball sport played between two teams of eleven players each. Originating in England in the 16th century, it has grown into one of the most followed games worldwide, especially in countries such as India, Australia, England, Pakistan, South Africa, and the West Indies.

# The game is played on a circular or oval grass field with a 22-yard pitch at the center. Each team takes turns to bat and bowl (field). The batting side tries to score runs by hitting the ball and running between the wickets, while the bowling/fielding side tries to limit runs and dismiss batters."""

loader = PyPDFLoader('name.pdf')

# We can also load documents : 
docs = loader.load()    

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator='' 
)

result = splitter.split_documents(docs)

print(result)