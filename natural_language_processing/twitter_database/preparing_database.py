import pickle
from pathlib import Path

pickle_database_path = Path(__file__).parent / 'twitter_database.pkl'

with open(pickle_database_path, 'rb') as f:
    twitter_database = pickle.load(f)
    
final_database = []
for tweet_text, sentiment in zip(twitter_database['tweet_text'], twitter_database['sentiment']):
    if sentiment == 1:
        sentiment_dict = {'POSITIVE': True, 'NEGATIVE': False}
    else: 
        sentiment_dict = {'POSITIVE': False, 'NEGATIVE': True}
    final_database.append([tweet_text, sentiment_dict.copy()])
    
pickle_final_database_path = Path(__file__).parent / 'twitter_final_database.pkl'

with open(pickle_final_database_path, 'wb') as f:
    pickle.dump(final_database, f)