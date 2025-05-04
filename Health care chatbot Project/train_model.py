from chatbot import HealthChatbotModel

model = HealthChatbotModel()
model.load_data('data/train_data_chatbot.csv')  
model.save_model('data/chatbot_model.pkl')

print(" Model trained and saved successfully!")