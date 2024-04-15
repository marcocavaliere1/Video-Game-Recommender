from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse as sp

app = Flask(__name__)

df = pd.read_csv('./Data/games_clean.csv')

# vectorize summary feature
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['summary'])

# one-hot encode categorical features (platforms, genres, game_modes)
mlb = MultiLabelBinarizer()
platforms_encoded = mlb.fit_transform(df['platforms'])
genres_encoded = mlb.fit_transform(df['genres'])
game_modes_encoded = mlb.fit_transform(df['game_modes'])

# combine the features using sp.hstack so we can put the arrays/matrices together
combined_features = sp.hstack((tfidf_matrix, platforms_encoded, genres_encoded, game_modes_encoded), format='csr')

# nearest Neighbors Model
# number of neighbors
nn_model = NearestNeighbors(n_neighbors=10, algorithm='auto')
nn_model.fit(combined_features)

def recommend_game(game_title, k=10):
    game_title = game_title.lower() # convert game title to lower case so that casing will not matter when inputing game
    # find the index of the input game
    try:
        # find the index of the input game
        game_index = df[df['name'].str.lower() == game_title].index[0]
        
        # get the combined feature vector for the input game
        input_features = sp.hstack((tfidf_vectorizer.transform([df.iloc[game_index]['summary']]), 
                                    platforms_encoded[game_index].reshape(1, -1),  
                                    genres_encoded[game_index].reshape(1, -1),     
                                    game_modes_encoded[game_index].reshape(1, -1)))  
        
        # let's say input is call of duty, game was recommending different call of duty's, 
        # so I need more k's so I can eliminate the ones with similar names in my list if needed, and still get 10 recs.
        n_neighbors = max(2 * k, 100)  
        
        # find the nearest neighbors
        distances, indices = nn_model.kneighbors(input_features, n_neighbors=n_neighbors)
        
        # list for recs.
        base_game_recommendations = {}
        
        # go through the list 
        for dist, idx in zip(distances.squeeze(), indices.squeeze()):
            game_name = df.iloc[idx]['name']
            
            base_name = game_name.split(':')[0].strip().lower()
            
            # to exclude games with similar titles
            if game_title not in game_name.lower() and game_name not in base_game_recommendations.values():
                base_game_recommendations[base_name] = {
                    'name': game_name,
                    'url': df.iloc[idx]['url']
                }
                
        recommended_games = list(base_game_recommendations.values())[:k]  # return k unique recommendations
        return recommended_games
    
    except IndexError:
        # Handle the case when the game title is not found in the database
        return None


# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        game_title = request.form['game_title']
        recommendations = recommend_game(game_title)
        return render_template('recommendations.html', game_title=game_title, recommendations=recommendations)
    else:
        error_message = f"Sorry, the game '{game_title}' is not in the database. Please check the spelling or try another game."
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)



