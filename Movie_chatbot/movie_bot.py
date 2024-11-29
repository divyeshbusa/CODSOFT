import requests
from config import TMDB_API_KEY

BASE_URL = "https://api.themoviedb.org/3"

def get_movie_details(movie_name):
    search_url = f"{BASE_URL}/search/movie"
    response = requests.get(search_url, params={"api_key": TMDB_API_KEY, "query": movie_name})
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            return {
                "title": movie['title'],
                "release_date": movie['release_date'],
                "overview": movie['overview']
            }
        else:
            return "Movie not found."
    else:
        return "Error fetching data."

def get_director(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/credits"
    response = requests.get(url, params={"api_key": TMDB_API_KEY})
    if response.status_code == 200:
        data = response.json()
        for crew in data['crew']:
            if crew['job'] == 'Director':
                return crew['name']
        return "Director not found."
    else:
        return "Error fetching data."

def get_movie_id(movie_name):
    search_url = f"{BASE_URL}/search/movie"
    response = requests.get(search_url, params={"api_key": TMDB_API_KEY, "query": movie_name})
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['id']
        else:
            return None
    return None

def suggest_movies(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/recommendations"
    response = requests.get(url, params={"api_key": TMDB_API_KEY})
    if response.status_code == 200:
        data = response.json()
        recommendations = [movie['title'] for movie in data.get('results', [])]
        return recommendations if recommendations else "No suggestions available."
    return "Error fetching suggestions."



def main():
    print("Welcome to the Movie Bot!")
    while True:
        query = input("Ask me about a movie or type 'exit' to quit: ").lower()
        if query.startswith("director of"):
            movie_name = query.split("director of")[-1].strip()
            movie_id = get_movie_id(movie_name)
            if movie_id:
                print(f"The director of {movie_name} is {get_director(movie_id)}.")
            else:
                print("Movie not found.")
        elif query.startswith("recommend movies like"):
            movie_name = query.split("recommend movies like")[-1].strip()
            movie_id = get_movie_id(movie_name)
            if movie_id:
                suggestions = suggest_movies(movie_id)
                print(f"Movies similar to {movie_name}: {', '.join(suggestions)}")
            else:
                print("Movie not found.")
        elif query == "exit":
            print("Goodbye!")
            break
        else:
            print("I can only answer questions about directors or suggestions for now.")

main()

