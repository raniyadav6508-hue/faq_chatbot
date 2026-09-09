import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load FAQ data
data = pd.read_csv('faqs.csv')
questions = data['question'].tolist()
answers = data['answer'].tolist()

print("FAQ data loaded successfully!")
print(f"Total FAQs: {len(questions)}")

# Step 2: Text preprocessing function
def preprocess(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'[^\w\s]', '', text)           # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra spaces
    return text

# Preprocess all FAQ questions
processed_questions = [preprocess(q) for q in questions]

print("\nSample preprocessed question:")
print(f"Original : {questions[0]}")
print(f"Processed: {processed_questions[0]}")
# Step 3: Convert questions into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(processed_questions)

# Step 4: Function to get chatbot response
def get_response(user_question):
    processed_input = preprocess(user_question)
    input_vector = vectorizer.transform([processed_input])
    
    similarities = cosine_similarity(input_vector, question_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]
    
    if best_score < 0.4:  # threshold — agar match zyada kamzor hai
        return "Sorry, I couldn't find a good answer for that. Please rephrase your question."
    
    return answers[best_match_index]

# Step 5: Test it
test_question = "What time does college open?"
print(f"\nUser: {test_question}")
print(f"Bot: {get_response(test_question)}")
# Step 6: Interactive chat loop
print("\n--- FAQ Chatbot ---")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    response = get_response(user_input)
    print(f"Bot: {response}")
