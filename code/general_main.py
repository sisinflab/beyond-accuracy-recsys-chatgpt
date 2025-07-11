import json
import time
import os
import warnings
from sys import exit

import httpx
import openai
from open_ai_model import Open_AI

from utils.read_movielens import merge_titles_with_movies
from utils.utils import *
from utils.subset_creator import *

from datetime import datetime

# from external.elliot.run import run_experiment

from utils.utils import SearchCache

warnings.filterwarnings("ignore", message="Setuptools is replacing distutils.")

def compute_time(start_time, end_time):
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Format the elapsed time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Display the formatted time
    formatted_time = f"{hours}h {minutes}min {seconds}s"

    return formatted_time

def send_message(model, dataset, exp_type, checkpoint_dir):
    """
    Send message to LLM
    :param model: 'gpt-3.5-turbo-1106'
    :param dataset: 'facebook_book', 'hetrec2011_lastfm_2k', 'ml_small_2018'
    :param checkpoint_dir: the name of the directory for saving the checkpoints
    :param exp_type: 'EXP_1' for Recommendation, 'EXP_2' for Re-rank MostPop, 'EXP_3' for Re-rank UserKNN
    """

    chat_gpt = None
    train = None

    if model == 'gpt-3.5-turbo-1106':
        chat_gpt = Open_AI('gpt-3.5-turbo-1106')
    else:
        print('No model name found')
        exit()

    # Retrieve train data
    if dataset == 'facebook_book':
        train = pd.read_csv('../data/dataset/facebook_book/trainingset_with_name.tsv',
                            sep="\t", header=None,
                            names=['userId', 'bookId', 'rating', 'name'],
                            usecols=['userId', 'bookId', 'rating', 'name'])
        # Initialize variable for the request
        utils = Utils(train)
    elif dataset == 'hetrec2011_lastfm_2k':
        train = pd.read_csv('../data/dataset/hetrec2011_lastfm_2k/splitting/0/train_with_name.tsv', sep="\t",
                            header=None, names=['userId', 'artistId', 'weight', 'name', 'url', 'pictureURL'],
                            usecols=['userId', 'artistId', 'weight', 'name'])
        # Initialize variable for the request
        utils = Utils(train)
    elif dataset == 'ml_small_2018':
        # Retrieve ratings and movies information
        if exp_type == 'EXP_1':
            ratings = pd.read_csv('../data/dataset/ml_small_2018/splitting/0/subset_train_230.tsv', sep='\t',
                              header=None, names=['userId', 'movieId', 'rating'])
        else:
            ratings = pd.read_csv('../data/dataset/ml_small_2018/splitting/0/subset_train_200.tsv', sep='\t',
                                  header=None, names=['userId', 'movieId', 'rating'])
        movies = pd.read_csv('../data/dataset/ml_small_2018/movies.csv', sep=',', header=0,
                             names=['movieId', 'title', 'genres'], usecols=['movieId', 'title'])
        # Create the ratings dataframe with the movies title
        train = merge_titles_with_movies(ratings, movies)
        # Initialize variable for the request
        utils = Utils(train)
    else:
        print('No dataset name found')
        exit()

    last_user_checkpoint = None
    max_user_id = -1
    highest_user_file = None

    # Iterate over the directory entries and find the file with the highest userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint.txt")])
                if user_id > max_user_id:
                    max_user_id = user_id
                    highest_user_file = entry.path

    if os.path.exists(highest_user_file):
        last_user_checkpoint = highest_user_file

    for user in train['userId'].unique():
        # Skip users until the last checkpoint is reached
        if last_user_checkpoint is not None and user <= max_user_id:
            continue

        message = ''
        # Generate the message for ChatGPT
        if exp_type == 'EXP_1':
            if dataset == 'facebook_book':
                message = utils.book_read_by_user(user)
            elif dataset == 'hetrec2011_lastfm_2k':
                message = utils.artists_listened_by_user(user)
            elif dataset == 'ml_small_2018':
                message = utils.movies_rated_by_user(user)
        if exp_type == 'EXP_2':
            if dataset == 'facebook_book':
                message = utils.rerank_by_user_profile_facebook(user)
            elif dataset == 'hetrec2011_lastfm_2k':
                message = utils.rerank_by_user_profile_hetrec(user)
            elif dataset == 'ml_small_2018':
                message = utils.rerank_by_user_profile(user)
        if exp_type == 'EXP_3':
            result = []
            if dataset == 'facebook_book':
                message = utils.rerank_by_similar_user_profile_facebook(user, result)
            elif dataset == 'hetrec2011_lastfm_2k':
                message = utils.rerank_by_similar_user_profile_hetrec(user, result)
            elif dataset == 'ml_small_2018':
                message = utils.rerank_by_similar_user_profile(user, result)

        # Send message to ChatGPT
        response = chat_gpt.request(message, exp_type)

        # Checkpoint foreach user
        checkpoint_file = os.path.join(checkpoint_dir, f'user_{user}_checkpoint.txt')
        with open(checkpoint_file, 'w') as f:
            # f.write(response['choices'][0]['message']['content'])
            f.write(response.choices[0].message.content)

        if exp_type == 'EXP_3':
            # Save top50 no rerank for each user
            with open('../data/dataset/' + dataset + '/v2/' + model + '/output_exp_3_no_rerank_userknn.tsv', 'a',
                      newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                for row in result:
                    writer.writerow(row)


def convert_response_to_tsv(dataset, checkpoint_dir, output_path, cache, use_llm_similarity_check):
    recommendations = None
    result = {}

    # Iterate over the directory entries and find the userId
    with os.scandir(checkpoint_dir) as entries:
        for entry in entries:
            if entry.name.startswith("user_") and entry.name.endswith("_checkpoint.txt"):
                user_id = int(entry.name[len("user_"):-len("_checkpoint.txt")])
                with open(entry, 'r') as file:
                    response = file.read()
                # Create the dictionary of recommendations
                if dataset == 'facebook_book':
                    recommendations = parse_item_recommendations(response, user_id, recommendations)
                elif dataset == 'hetrec2011_lastfm_2k':
                    recommendations = parse_artist_recommendations(response, user_id, recommendations)
                elif dataset == 'ml_small_2018':
                    recommendations = parse_movie_recommendations(response, user_id, recommendations)

    # Check for ids from the dataset
    if dataset == 'facebook_book':
        result = optimized_search_item_v2(recommendations,
                                          items_df=pd.read_csv('../data/dataset/facebook_book/books.tsv',
                                                               sep='\t', names=['id', 'name']),
                                          use_llm_similarity_check=use_llm_similarity_check, cache=cache)

    elif dataset == 'hetrec2011_lastfm_2k':
        result = optimized_search_item_v2(recommendations, items_df=pd.read_csv('../data/dataset/hetrec2011_lastfm_2k/artists.dat',
                                                                                sep='\t', header=0, names=['id', 'name', 'url', 'pictureURL'],
                                                                                usecols=['id', 'name']),
                                          use_llm_similarity_check=use_llm_similarity_check, cache=cache)
    elif dataset == 'ml_small_2018':
        try:
            result = optimized_search_item_v2(recommendations, items_df=pd.read_csv('../data/dataset/ml_small_2018/movies.csv',
                                                                                 sep=',', header=0, names=['id', 'name', 'genres'],
                                                                                    usecols=['id', 'name']),
                                           use_llm_similarity_check=use_llm_similarity_check, cache=cache)
        except KeyError as e:
            print("Key error: ", e)
    #save_result(result, output_path)
    optimized_save_result(result, output_path)

def main():
    """
    @dataset: 'facebook_book', 'hetrec2011_lastfm_2k', 'ml_small_2018'
    @model: 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'
    """

    for dataset in ['facebook_book', 'hetrec2011_lastfm_2k', 'ml_small_2018']: # 'facebook_book', 'hetrec2011_lastfm_2k', 'ml_small_2018'
        print("Starting Evaluation: "+dataset+"\n")
        cache = SearchCache()
        for exp_type in ['EXP_1', 'EXP_2', 'EXP_3']: # 'EXP_1', 'EXP_2', 'EXP_3'
            print(exp_type)
            try:
                model = 'gpt-3.5-turbo-1106' # 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'

                # Create the dir tree
                checkpoint_dir = '../data/dataset/' + dataset + '/v2/' + model + '/' + exp_type
                if not os.path.exists(checkpoint_dir):
                    try:
                        os.makedirs(checkpoint_dir)
                        with open(checkpoint_dir+'/user_0_checkpoint.txt', 'w') as fp:
                            pass
                        print(f"Directory '{checkpoint_dir}' created successfully.")
                    except OSError as error:
                        print(f"Error occurred while creating directory: {error}")
                else:
                    print(f"Directory '{checkpoint_dir}' already exists.")

                #send_message(model=model, dataset=dataset, exp_type=exp_type, checkpoint_dir=checkpoint_dir)

            except openai.APITimeoutError as e:
                print("Request time out: {}".format(e))
                time.sleep(20)
                main()
            except openai.RateLimitError as e:
                print("API rate limit exceeded: {}".format(e))
                time.sleep(20)
                main()
            except openai.APIConnectionError as e:
                print("API connection error: {}".format(e))
                time.sleep(20)
                main()
            except json.JSONDecodeError as e:
                print("JSONDecodeError: {}".format(e))
                time.sleep(20)
                main()
            except openai.APIError as e:
                print("HTTP code 502 from API: {}".format(e))
                time.sleep(20)
                main()
            except httpx.HTTPStatusError as e:
                print("HTTPStatusError: {}".format(e))
                time.sleep(20)
                main()

            print("Starting converting to TSV " + dataset)
            convert_response_to_tsv(dataset=dataset, checkpoint_dir=checkpoint_dir, output_path=checkpoint_dir,
                                    cache=cache, use_llm_similarity_check = True)
            print("Conversion "+dataset+" "+exp_type+" finished.\n")
            pass
        cache.clear_caches()




if __name__ == '__main__':
    # Record the starting time
    start_time = time.time()
    main()
    # Record the ending time
    end_time = time.time()
    print(compute_time(start_time, end_time))
    pass
