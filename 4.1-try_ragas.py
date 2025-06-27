import os
from datasets import Dataset
from dotenv import load_dotenv


from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# .env dosyasındaki değişkenleri yükle
load_dotenv()


gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


ragas_llm = LangchainLLMWrapper(gemini_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)


faithfulness.llm = ragas_llm
answer_correctness.llm = ragas_llm
answer_correctness.embeddings = ragas_embeddings


data_sample = {
   
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
   
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
   
    'contexts': [['The first AFL-NFL World Championship Game was an American footbal game played on January 15, 1967'], ['The New England Patriots have won the Super Bowl a record six times']],
   
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}
dataset = Dataset.from_dict(data_sample)

score = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_correctness
    ]
)

df = score.to_pandas()
df.to_csv('ragas_evaluation_final.csv', index=False)

print("Değerlendirme tamamlandı")
print(df)