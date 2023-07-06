import gradio as gr
import requests
import numpy as np 
import pandas as pd 
from PIL import Image

# api_key = "4e45e5b0" 

# A function that takes a movie name and returns its poster image as a numpy array
def get_poster(movie):
    api_key = "4e45e5b0" 
    base_url = "http://www.omdbapi.com/"
    
    params = {"apikey": api_key , "t": movie}
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if data['Response'] == 'True': # Check if the response is successful
        # Open the image from the url
        poster_image = Image.open(requests.get(data['Poster'], stream=True).raw)
        # Convert the image to a numpy array
        poster_array = np.array(poster_image)
        return poster_array 
    
    else:
        return np.zeros((500, 500, 3))

# A function that takes a movie name and returns its meta data
def get_data(movie):
    api_key = "4e45e5b0" 
    base_url = "http://www.omdbapi.com/"
    
    params = {"apikey": api_key , "t": movie}
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if data['Response'] == 'True': # Check if the response is successful  
        poster = data["Poster"]
        title = data["Title"]
        director = data["Director"]
        cast = data["Actors"]
        genres = data["Genre"]
        rating = data["imdbRating"]
        # Return a dictionary with the information
        return {
                    
            "poster": poster,
            "title": title,
            "director": director,
            "cast": cast,
            "genres": genres,
            "rating": rating
               }

def get_recommendations(input_list):
    movie_names = ["The Matrix", "The Shawshank Redemption", "The Godfather", "The Dark Knight", "Inception"]
    movies_data = [get_data(movie) for movie in movie_names]
    
    movie_posters = [get_poster(movie) for movie in movie_names]
    
    
    return movie_names, movie_posters
# HTML table for recommendation section 
def generate_table(movies, posters):
    html_code = ""
    # Add the table tag and style attributes
    html_code += "<table style='width:100%; border: 1px solid black; text-align: center;'>"
    
    for i in range(len(movies)):
        movie_name = movies[i]
        poster_array = posters[i]
        movie_data = get_data(movie_name)
        
        # Extract the information from the dictionary
        poster_url = movie_data["poster"]
        title = movie_data["title"]
        director = movie_data["director"]
        cast = movie_data["cast"]
        genres = movie_data["genres"]
        rating = movie_data["rating"]
        
        # Add a table row tag for each movie
        html_code += "<tr>"
        # Add a table cell tag with the poster image as an img tag
        html_code += f"<td><img src='{poster_url}' height='400' width='300'></td>"
        # Add a table cell tag with the movie information as a paragraph tag
        html_code += f"<td><p><b>Title:</b> {title}</p><p><b>Director:</b> {director}</p><p><b>Cast:</b> {cast}</p><p><b>Genres:</b> {genres}</p><p><b>Rating:</b> {rating}</p></td>"
        # Close the table row tag
        html_code += "</tr>"

    # Close the table tag 
    html_code += "</table>"
    
    return html_cod

user_input = {}

def display_movie(movie, rating):
    
    global user_input 
    user_input[movie] = rating 
    poster = get_poster(movie)
    
    if len(user_input) > 5:
        # Get the recommended movies from the input 
        r_movies, r_posters = get_recommendations(user_input) 
        
        # Create a list with a list of HTML strings with information 
        html_code =  generate_table(r_movies, r_posters)
        
        user_input = {}
        # Return the output 
        return f"Your movies are ready!\nPlease check the recommendations below.", np.zeros((500, 500, 3)), html_code
    
    else:
        
        # Return the input movie name and poster 
        return f"You entered {movie} with rating {rating}", poster, ""

iface = gr.Interface(
    
    fn= display_movie, 
    inputs= [gr.Textbox(label="Enter a movie name"), gr.Slider(minimum=0, maximum=5, step=1, label="Rate the movie")],
    outputs= [gr.Textbox(label="Output", min_width=200), gr.components.Image(label="Poster", height=400, width=300), gr.components.HTML(label="Recommendations", height=400)],
    
    live= False
    )

iface.launch()