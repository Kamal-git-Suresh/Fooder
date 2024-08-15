import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import difflib
from nltk.corpus import wordnet
import pickle

# Load the recipe data
recipe_data = pd.read_csv(r'D:\Code\Mini-Project\recipe_details_clean.csv')

# Fill missing values in the dataset
recipe_data = recipe_data.fillna('')
#print(0)

bert = SentenceTransformer('bert-base-nli-mean-tokens')
#print('1')
# Preprocess the data
recipe_data['ingredients_text'] = recipe_data['Ingredients'].apply(lambda x: ', '.join(str(x).split(',')) if isinstance(x, str) else '')
recipe_data['description_text'] = recipe_data['Description']
recipe_data['features'] = recipe_data['ingredients_text'] + ' ' + recipe_data['description_text']
#print('2')
def find_synonyms(word):
    synonims = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonims.add(lemma.name())
        return synonims
    
# Create a TF-IDF matrix
tfidf = TfidfVectorizer()#stop_words='english')
tfidf_matrix = tfidf.fit_transform(recipe_data['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#print('3')
#bert_encodings = bert.encode(recipe_data['features'])
#with open('bert_encodings.plk','rb') as bert_enc:
    #pickle.dump(bert_encodings, bert_enc)
#    bert_encodings = pickle.load(bert_enc)
#print('4')
#print(bert_encodings)
# Calculate the cosine similarity matrix
#print('5')

#similarity_scores = cosine_similarity(bert_encodings, bert.encode(['Broccoli & Cheese Soup','Keema Potato Casserole','Macaroni Alfredo']))
#print('6')
#print(similarity_scores[0].shape)
#print('7')

'''def get_top_n_recipes_bert(dish_titles, cosine_sim=similarity_scores, topn=10):
    indices = []
    for title in dish_titles: 
        idx = recipe_data[recipe_data['Name'] == title].index
        indices.extend(idx)

    sim_scores = []
    for idx in indices:
        sim_scores.extend(list(enumerate(cosine_sim[idx])))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:topn+1]
    recipe_indices = [i[0] for i in sim_scores]
'''
    # Print row number and name of the dishes
    #for idx in recipe_indices:
    #    print(f"Row Number: {idx}, Dish Name: {recipe_data.iloc[idx]['Name']}")
# Define a function to get the top N most similar recipes for multiple dishes
def get_top_n_recipes_for_multiple(dish_titles, cosine_sim=cosine_sim, topn=10):
    indices = []
    for title in dish_titles: 
        idx = recipe_data[recipe_data['Name'] == title].index
        indices.extend(idx)

    sim_scores = []
    for idx in indices:
        sim_scores.extend(list(enumerate(cosine_sim[idx])))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:topn+1]
    recipe_indices = [i[0] for i in sim_scores]
    reccomendations = []
    # Print row number and name of the dishes
    for idx in recipe_indices:
        #print(f"Row Number: {idx}, Dish Name: {recipe_data.iloc[idx]['Name']}")
        reccomendations.append(recipe_data.iloc[idx]['Name'])
    return reccomendations


given_ingredients = {'garlic', 'parmesan', 'parsley', 'butter', 'pepper', 'cream', 'salt', 'milk', 'macaroni', 'white', 'powder', 'cloves', 'cheese', 'capsicums','grams', 'jalapenos', 'oil', 'olive', 'pasta', 'penne', 'red', 'salt', 'tomatoes','bananas', 'chilli', 'coriander', 'cumin', 'fenugreek', 'leaves','mustard', 'oil', 'powder', 'salt', 'sev', 'turmeric'}
matching_recipes = []

for index, row in recipe_data.iterrows():
    # Split the ingredients string into a list
    row_ingredients = set(row['Ingredients'].split())
    #    print(row_ingredients)
    #    print(given_ingredients)
    # Remove any extra spaces from the ingredients
    #row_ingredients = [ingredient.strip() for ingredient in row_ingredients]
    # Check if all the ingredients in the list are in the current recipe
    #if all(row_ingredients) in given_ingredients:
    for item in row_ingredients:
        if item not in given_ingredients:
            break
        else:
            matching_recipes.append(row['Name'])
            
    #if all(ingredient in row_ingredients for ingredient in given_ingredients):
        # If they are, add the recipe name to the list
        
#print(matching_recipes)
# Example usage with multiple dishes
dish_titles = ['Broccoli & Cheese Soup','Keema Potato Casserole','Macaroni Alfredo']
mask = recipe_data['Ingredients'].apply(lambda x: any(ing in x for ing in given_ingredients))
#recipe_names = recipe_data[mask]['Name']
#print(recipe_names)

#dish_titles = [title for title in difflib.get_close_matches(dish_titles)]
#temp_recs = get_top_n_recipes_for_multiple(dish_titles, topn = 6)
#print(temp_recs)

#print(get_top_n_recipes_bert(dish_titles))
#print(recipe_names)
def get_reco(given_ingredients, dish_titles):
    temp_recs = get_top_n_recipes_for_multiple(dish_titles, topn = 10)
    if len(temp_recs) != 0:
        return set(temp_recs) - set(dish_titles)
    else:
        return 'There are no dishes that can be made with the available ingredients'
    
def hybrid(target, similar):
    temp_recs = get_top_n_recipes_for_multiple(target, topn = 10)
    
    return set(temp_recs)
    