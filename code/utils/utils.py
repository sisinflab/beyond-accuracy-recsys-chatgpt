import tiktoken
import pandas as pd
import csv
import difflib
import re

from utils.load_userknn import UserKnnLoader, search_50_unseen_movies, search_50_unlisten_artist, search_50_unread_book
from utils.facebook_book_utils import get_author_name

# Relative import
# from .load_userknn import UserKnnLoader, search_50_unseen_movies, search_50_unlisten_artist, search_50_unread_book
# from .facebook_book_utils import get_author_name


from utils.llm_utils import check_item_similarity_mistral_7b, check_item_similarity_mistral_7b_v2
import unicodedata

def test_chat_gpt_output():
    output_str = """
            Based on the user's preferences, here are the top 50 movie recommendations:

            1. Pulp Fiction (1994)
            2. Forrest Gump (1994)
            3. The Godfather (1972)
            4. The Godfather: Part II (1974)
            5. The Silence of the Lambs (1991)
            6. Schindler's List (1993)
            7. The Shawshank Redemption (1994)
            8. Goodfellas (1990)
            9. Saving Private Ryan (1998)
            10. The Matrix (1999)
            11. American Beauty (1999)
            12. Fight Club (1999)
            13. The Green Mile (1999)
            14. Gladiator (2000)
            15. Memento (2000)
            16. A Beautiful Mind (2001)
            17. The Lord of the Rings: The Fellowship of the Ring (2001)
            18. The Lord of the Rings: The Two Towers (2002)
            19. The Lord of the Rings: The Return of the King (2003)
            20. Kill Bill: Vol. 2 (2004)
            21. Million Dollar Baby (2004)
            22. Crash (2004)
            23. Batman Begins (2005)
            24. The Departed (2006)
            25. No Country for Old Men (2007)
            26. There Will Be Blood (2007)
            27. The Dark Knight (2008)
            28. Gran Torino (2008)
            29. Up (2009)
            30. Avatar (2009)
            31. The Hurt Locker (2009)
            32. Toy Story 3 (2010)
            33. The Social Network (2010)
            34. Black Swan (2010)
            35. True Grit (2010)
            36. The King's Speech (2010)
            37. Moneyball (2011)
            38. The Artist (2011)
            39. Hugo (2011)
            40. The Descendants (2011)
            41. Argo (2012)
            42. Lincoln (2012)
            43. Life of Pi (2012)
            44. Silver Linings Playbook (2012)
            45. Gravity (2013)
            46. 12 Years a Slave (2013)
            47. Her (2013)
            48. Captain Phillips (2013)
            49. The Grand Budapest Hotel (2014)
            50. Birdman (2014)
            """
    # Create a dictionary of {user: [(title1, rank), ..., (titleN, rank)]}
    recommendations = parse_movie_recommendations(output_str, '2')
    recommendations = parse_movie_recommendations(output_str, '3', recommendations)

    # Search for movie IDs and write to TSV file
    result = search_movies(recommendations)
    # Open the file in write mode and specify the delimiter as '\t' for TSV
    save_result(result)

"""
    MovieLens Utils
"""
# Define function to search for movie IDs based on title similarity and return list of tuples
def search_movies(user_dict):
    # Load movielens dataset and create a dictionary mapping movie titles to movie IDs
    movies_df = pd.read_csv('../data/dataset/ml_small_2018/movies.csv', sep=',', usecols=['movieId', 'title'])
    movies_dict = dict(zip(movies_df.title, movies_df.movieId))
    # Create a List of (user, title, movie_id, rank) for each element
    result = []
    for user, title_rank_list in user_dict.items():
        for title, rank in title_rank_list:
            closest_match = difflib.get_close_matches(title, movies_dict.keys(), n=1)
            if closest_match:
                movie_id = movies_dict[closest_match[0]]
                result.append((user, title, movie_id, rank))
    return result
def save_result(result, output_path):
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        # Write the header row to the file
        header = ['userId', 'title', 'itemId', 'rank']
        writer.writerow(header)

        # Write each row of the data to the file
        for row in result:
            writer.writerow(row)

"""
    Utils for Saving Results
"""
def write_csv(filename, header, data_rows):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(header)

        # Check if data_rows is a dictionary
        if isinstance(data_rows, dict):
            # Flatten the dictionary values into a list
            data_rows = [row for rows in data_rows.values() for row in rows]

        writer.writerows(data_rows)

def calculate_stats(data):
    return {
        'min_length': min(len(lst) for lst in data.values()),
        'max_length': max(len(lst) for lst in data.values()),
        'user_id_list_less_20': {lst[0][0] for lst in data.values() if len(lst) < 20}
    }

def optimized_save_result(result, output_path):
    # Save results with and without duplicates
    write_csv(output_path + "_w_duplicate.tsv", ['userId', 'chatGPT_title', 'dataset_name', 'itemId', 'rank'], result[0])
    write_csv(output_path + "_no_duplicate.tsv", ['userId', 'chatGPT_title', 'dataset_name', 'itemId', 'rank'], result[1][0])

    # Calculate stats
    stats_w_duplicate = calculate_stats(result[0])
    stats_no_duplicate = calculate_stats(result[1][0])

    total_users = len(result[0])
    log_content = f"""
    number of unique hallucination: {len(result[2])}
    number of user with at least one hallucination: {len(result[3])} on {total_users} total users
    number of user with at least one repetition hallucination: {len(result[1][1])} on {total_users} total users
    min item list w duplicate: {stats_w_duplicate['min_length']}
    max item list w duplicate: {stats_w_duplicate['max_length']}
    number of user with item list len less then 20: {len(stats_w_duplicate['user_id_list_less_20'])} on {total_users} total users
    min item list w/o duplicate: {stats_no_duplicate['min_length']}
    max item list w/o duplicate: {stats_no_duplicate['max_length']}
    number of user with item list len less then 20: {len(stats_no_duplicate['user_id_list_less_20'])} on {total_users} total users
    """

    # Save log
    with open(output_path + ".log", 'w') as file:
        file.write(log_content.strip())

    # Save the list of hallucination item
    write_csv(output_path + "_list_of_hallucination.tsv", ['item title'], [(x,) for x in result[2].keys()])



def extract_name_year(movie_string):
    # Using regular expressions to extract the name and year
    match = re.match(r"(.+) \((\d+)\)", movie_string)
    if match:
        name = match.group(1)
        year = int(match.group(2))
    else:
        name, year = None, None

    return name, year
def parse_movie_recommendations(output_str, user, recommendations=None):
    if recommendations is None:
        recommendations = {user: []}
    if user not in recommendations:
        recommendations[user] = []
    for line in output_str.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            rank, movie = line.split('.', 1)
            movie = movie.strip()
            recommendations[user].append((movie, rank))
    return recommendations

"""
    HetRec Utils
"""
def search_artist(user_dict):
    # Load movielens dataset and create a dictionary mapping movie titles to movie IDs
    artist_df = pd.read_csv('../data/dataset/hetrec2011_lastfm_2k/artists.dat', sep="\t")
    artist_dict = dict(zip(artist_df.name, artist_df.id))
    # Create a List of (user, title, movie_id, rank) for each element
    result = []
    for user, title_rank_list in user_dict.items():
        for title, rank in title_rank_list:
            closest_match = difflib.get_close_matches(title, artist_dict.keys(), n=1)
            if closest_match:
                artist_id = artist_dict[closest_match[0]]
                result.append((user, title, artist_id, rank))
    return result
def save_artist_result(result):
    with open('../data/dataset/hetrec2011_lastfm_2k/gpt-3.5-turbo-1106/output_rec_artist_rerank_mostpop.tsv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        # Write the header row to the file
        header = ['userId', 'artist', 'artistId', 'rank']
        writer.writerow(header)

        # Write each row of the data to the file
        for row in result:
            writer.writerow(row)
def parse_artist_recommendations(output_str, user, recommendations=None):
    if recommendations is None:
        recommendations = {user: []}
    if user not in recommendations:
        recommendations[user] = []
    for line in output_str.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            rank, artist = line.split('.', 1)
            artist = artist.strip()
            recommendations[user].append((artist, rank))
    return recommendations

"""
    Facebook Utils
"""
def parse_item_recommendations(output_str, user, recommendations=None):
    if recommendations is None:
        recommendations = {user: []}
    if user not in recommendations:
        recommendations[user] = []
    for line in output_str.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            try:
                rank, item = line.split('.', 1)
                if len(item.split(' by ', 1)) == 2:
                    book, author = item.split(' by ', 1)
                elif len(item.split(' by ', 1)) == 1:
                    book = item.split(' by ', 1)
                recommendations[user].append((book, rank))
            except ValueError:
                try:
                    while len(recommendations[user]) < 50:
                        recommendations[user].append(
                            (recommendations[user][len(recommendations[user]) - 1][0], str(len(recommendations[user]) + 1)))
                except IndexError as e:
                    print(f"User {user}, {e}", e)
    return recommendations

"""
    Search for the corresponding item into the datasets (The close words) if exists
    otherwise, count as hallucination
"""
def search_item(user_dict, items_df):
    items_dict = dict(zip(items_df.name, items_df.id))
    # Create a List of (user, title, movie_id, rank) for each element
    result = []
    hallucination = []

    # Store item names outside the loops to avoid redundant calls
    item_names = list(items_dict.keys())
    # Pre-compile regex patterns for each item name
    regex_patterns = {item_name: re.compile(re.escape(item_name)) for item_name in item_names}

    for user, title_rank_list in user_dict.items():
        for title, rank in title_rank_list:
            closest_match = difflib.get_close_matches(title, item_names, n=3)
            if closest_match != []:
                score = difflib.SequenceMatcher(None, title, closest_match[0]).ratio()
                if score >= 0.7:
                    item_id = items_dict[closest_match[0]]
                    result.append((user, title, item_id, rank))
                else:
                    hallucination.append((user, title, rank))
            elif closest_match == []:
                # Search pattern of recommended item in any part of item_name in the dataset
                item_found = False
                for item_name in item_names:
                    pattern = regex_patterns[item_name]
                    if pattern.search(title):
                        item_id = items_dict[item_name]
                        result.append((user, title, item_id, rank))
                        item_found = True
                        break
                # If no item found, add hallucination
                if item_found == False:
                    hallucination.append((user, title, rank))
    return result, hallucination

# Function to remove all characters after the first "(" in a string
def remove_after_parenthesis(text):
    # Find the position of the first "("
    pos = text.find("(")
    # If "(" is found, return the substring up to "("
    if pos != -1:
        return text[:pos].strip()
    # If "(" is not found, return the original text
    return text

def advanced_compare_strings(str_1, str_2):
    # Function to normalize the string
    def normalize(s):
        # Lowercase and remove punctuation, except parentheses
        s = re.sub(r'[^\w\s()]', '', s.lower())
        # Remove additional content in parentheses
        s = re.sub(r'\(.*?\)', '', s)
        # Split into words and remove extra spaces
        words = s.split()
        # Move "The" from beginning to end if present
        if words and words[0] == 'the':
            words.append(words.pop(0))
        return set(words)

    # Normalize both strings
    normalized_str_1 = normalize(str_1)
    normalized_str_2 = normalize(str_2)

    # Compare the sets of words
    return normalized_str_1 == normalized_str_2

def advanced_compare_strings_v2(str_1, str_2):
    # Function to normalize the string
    def normalize(s):
        # Normalize unicode characters to remove accents
        s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
        # Lowercase and remove punctuation, except parentheses
        s = re.sub(r'[^\w\s()]', '', s.lower())
        # Remove additional content in parentheses
        s = re.sub(r'\(.*?\)', '', s)
        # Split into words and remove extra spaces
        words = s.split()
        # Move "The" from beginning to end if present
        if words and words[0] == 'the':
            words.append(words.pop(0))
        return set(words)

    # Normalize both strings
    normalized_str_1 = normalize(str_1)
    normalized_str_2 = normalize(str_2)

    # Compare the sets of words
    return normalized_str_1 == normalized_str_2

"""
    Optimized search for the corresponding item into the datasets (The close words) if exists
    otherwise, count as hallucination
"""
def OLD_optimized_search_item(user_dict, items_df, use_llm_similarity_check = False):
    # Convert DataFrame to dictionary for faster access
    items_dict = dict(zip(items_df.name, items_df.id))

    # Pre-compile regex patterns and store them in a dictionary for fast retrieval
    regex_patterns = {name: re.compile(re.escape(name)) for name in items_dict}

    # Initialize results and hallucination lists
    result = []
    hallucination = []

    # Caching for closest matches and hallucination to avoid repetitive difflib calls
    closest_matches_cache = {}
    hallucination_matches_cache = {}

    for user, title_rank_list in user_dict.items():
        for title, rank in title_rank_list:
            # Check cache first
            if title in closest_matches_cache:
                closest_match, score = closest_matches_cache[title]
            else:
                # Find closest matches and store the best match and its score in the cache
                closest_match = difflib.get_close_matches(title, items_dict.keys(), n=5)
                score = difflib.SequenceMatcher(None, title, closest_match[0]).ratio() if closest_match else 0
                closest_matches_cache[title] = (closest_match, score)

            if closest_match and score >= 0.7:
                item_id = items_dict[closest_match[0]]
                result.append((user, title, closest_match[0], item_id, rank))
            elif title in hallucination_matches_cache:
                hallucination.append((user, title, -1, rank))
                break
            else:
                # Check regex patterns for a match
                item_found = False
                # for item_name, pattern in regex_patterns.items():
                #     if pattern.search(title):
                #         item_id = items_dict[item_name]
                #         result.append((user, title, item_name, item_id, rank))
                #         item_found = True
                #         break
                for item_name in items_dict.keys():
                    if advanced_compare_strings_v2(title, item_name):
                        item_id = items_dict[item_name]
                        result.append((user, title, item_name, item_id, rank))
                        item_found = True
                        break
                if not item_found:
                    # for item_name in items_dict.keys():
                    #     if advanced_compare_strings_v2(title, item_name):
                    #         item_id = items_dict[item_name]
                    #         result.append((user, title, item_name, item_id, rank))
                    #         item_found = True
                    #         break
                    for item_name, pattern in regex_patterns.items():
                        if pattern.search(title):
                            item_id = items_dict[item_name]
                            result.append((user, title, item_name, item_id, rank))
                            item_found = True
                            break
                    if not item_found and use_llm_similarity_check and closest_match:
                        item_found = check_item_similarity_mistral_7b_v2(title, closest_match)
                        # if not item_found:
                        #     item_found = check_item_similarity_mistral_7b(title, items_dict.keys())
                        if item_found:
                            result.append((user, title,  item_found[0], items_dict[item_found[0]], rank))
                            closest_matches_cache[title] = (title, 0.7)
                            break

                # If no item found, add to hallucination
                if not item_found:
                    hallucination.append((user, title, -1, rank))
                    hallucination_matches_cache[title] = (title, -1)
    # results(user, chatGPT_title, dataset_name, rank)
    return result, hallucination, hallucination_matches_cache

def optimized_search_item(user_dict, items_df, use_llm_similarity_check = False, deep_search = False, verbose = True):
    # Convert DataFrame to dictionary for faster access
    items_dict = dict(zip(items_df.name, items_df.id))

    # Initialize results and hallucination lists
    result = {}
    hallucination = {}

    # Caching for closest matches and hallucination to avoid repetitive difflib calls
    closest_matches_cache = {}
    hallucination_matches_cache = {}

    for user, title_rank_list in user_dict.items():
        result[user] = []
        for title, rank in title_rank_list:
            # Check cache first
            title = str(title)
            if title in closest_matches_cache:
                closest_match, score = closest_matches_cache[title]
                item_id = items_dict[closest_match]
                result[user].append((user, title, closest_match, item_id, rank))
                if verbose:
                    print("Title: "+title+" IN closest_matches_cache")
                continue
            elif title in hallucination_matches_cache:
                hallucination[user] = []
                hallucination[user].append((user, title, -1, rank))
                if verbose:
                    print("Title: "+title+" IN hallucination_matches_cache")
                continue
            else:
                # Find closest matches and store the best match and its score in the cache
                closest_match = difflib.get_close_matches(title, items_dict.keys(), n=5)
                score = difflib.SequenceMatcher(None, title, closest_match[0]).ratio() if closest_match else 0

            if closest_match and score >= 0.7:
                item_id = items_dict[closest_match[0]]
                result[user].append((user, title, closest_match[0], item_id, rank))
                closest_matches_cache[title] = (closest_match[0], score)
                if verbose:
                    print("Title: "+title+" MATCH WITH "+closest_match[0]+" with score "+str(score))
            else:
                # Check regex patterns for a match
                item_found = False
                for item_name in items_dict.keys():
                    if advanced_compare_strings_v2(title, item_name):
                        item_id = items_dict[item_name]
                        result[user].append((user, title, item_name, item_id, rank))
                        item_found = True
                        if verbose:
                            print("REGEX: " + title + " MATCH WITH " + item_name)
                        break

                # Check similarity using LLM
                if not item_found and use_llm_similarity_check and closest_match:
                    item_found = check_item_similarity_mistral_7b_v2(title, closest_match)
                    if not item_found and deep_search:
                         item_found = check_item_similarity_mistral_7b_v2(title, items_dict.keys())
                    if item_found:
                        result[user].append((user, title, item_found, items_dict[item_found], rank))
                        closest_matches_cache[title] = (item_found, 0.7)
                        if verbose:
                            print("LLM: Found a match with " + title + " and " + item_found)
                        continue

                # If no item found, add to hallucination
                if not item_found:
                    hallucination[user] = []
                    hallucination[user].append((user, title, -1, rank))
                    hallucination_matches_cache[title] = (title, -1)
                    if verbose:
                        print("NOT FOUND: " + title)

    # results(user, chatGPT_title, dataset_name, rank)
    return result, hallucination, hallucination_matches_cache

def remove_duplicates(result_dict):
    new_result = {}
    user_id_w_repetition = set()
    for user_id, records in result_dict.items():
        seen = set()
        new_result[user_id] = []
        for record in records:
            if record[3] not in seen:  # Assuming record[3] is the item_id
                seen.add(record[3])
                new_result[user_id].append(record)
                user_id_w_repetition.add(record[0])
    return new_result, user_id_w_repetition

"""
    Optimized search for the corresponding item into the datasets (The close words) if exists
    otherwise, count as hallucination
    - Ottimizzare l'uso delle cache var per fare velocemente anche EXP_!/EXP_2/EXP_3 etc
"""
def optimized_search_item_v2(user_dict, items_df, cache, use_llm_similarity_check = False,
                             deep_search = False, verbose = True):
    # Convert DataFrame to dictionary and cache keys for faster access
    items_dict = dict(zip(items_df.name, items_df.id))

    # Initialize results and hallucination lists
    result = {}
    hallucination = {}

    for user, title_rank_list in user_dict.items():
        result[user] = []
        for title, rank in title_rank_list:
            title = str(title)

            # Check closest matches cache
            if title in cache.closest_matches_cache:
                closest_match, score = cache.closest_matches_cache[title]
                if closest_match in items_dict:
                    item_id = items_dict[closest_match]
                    result[user].append((user, title, closest_match, item_id, rank))
                    if verbose:
                        print(f"Title: {title} IN closest_matches_cache")
                continue

            # Check hallucination cache
            if title in cache.hallucination_matches_cache:
                hallucination[user] = []
                hallucination[user].append((user, title, -1, rank))
                if verbose:
                    print(f"Title: {title} IN hallucination_matches_cache")
                continue

            # Find closest matches and store the best match and its score in the cache
            closest_match = difflib.get_close_matches(title, items_dict.keys(), n=5)
            score = difflib.SequenceMatcher(None, title, closest_match[0]).ratio() if closest_match else 0

            if closest_match and score >= 0.7:
                item_id = items_dict[closest_match[0]]
                result[user].append((user, title, closest_match[0], item_id, rank))
                cache.closest_matches_cache[title] = (closest_match[0], score)
                if verbose:
                    print(f"Title: {title} MATCH WITH {closest_match[0]} with score {score}")
                continue

            # Check regex patterns for a match in the closest_match
            item_found = False
            if not item_found:
                for item_name in closest_match:
                    if advanced_compare_strings_v2(title, item_name):
                        item_id = items_dict[item_name]
                        result[user].append((user, title, item_name, item_id, rank))
                        item_found = True
                        cache.closest_matches_cache[title] = (item_name, 0.7)  # Cache this match
                        if verbose:
                            print(f"REGEX: {title} MATCH WITH {item_name}")
                        break

            # Check regex patterns for a match in the entire dataset
            if not item_found:
                for item_name in items_dict.keys():
                    if advanced_compare_strings_v2(title, item_name):
                        item_id = items_dict[item_name]
                        result[user].append((user, title, item_name, item_id, rank))
                        item_found = True
                        cache.closest_matches_cache[title] = (item_name, 0.7)  # Cache this match
                        if verbose:
                            print(f"REGEX: {title} MATCH WITH {item_name}")
                        break

            # Check similarity using LLM
            if use_llm_similarity_check and not item_found:
                if closest_match:
                    item_found = check_item_similarity_mistral_7b_v2(title, closest_match, verbose)
                if not item_found and deep_search:
                    item_found = check_item_similarity_mistral_7b_v2(title, items_dict.keys(), verbose)
                if item_found:
                    result[user].append((user, title, item_found, items_dict[item_found], rank))
                    cache.closest_matches_cache[title] = (item_found, 0.7)  # Cache this match
                    if verbose:
                        print(f"LLM: Found a match with {title} and {item_found}")
                    continue

            # If no item found, add to hallucination
            if not item_found:
                hallucination[user] = []
                hallucination[user].append((user, title, -1, rank))
                cache.hallucination_matches_cache[title] = (title, -1)
                if verbose:
                    print(f"NOT FOUND: {title}")

        if len(result[user]) == 0:
            del result[user]

    result_no_duplicate = remove_duplicates(result)

    # results(user, chatGPT_title, dataset_name, item_id, rank)
    return result, result_no_duplicate, cache.hallucination_matches_cache, hallucination

def save_author_result(result):
    with open('../data/dataset/hetrec2011_lastfm_2k/gpt-3.5-turbo-1106/output_rec_artist_rerank_mostpop.tsv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        # Write the header row to the file
        header = ['userId', 'artist', 'artistId', 'rank']
        writer.writerow(header)

        # Write each row of the data to the file
        for row in result:
            writer.writerow(row)



"""
    Utils for CSV - TSV
"""
def write_recommendations_to_tsv(recommendations, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['userId', 'title', 'rank'])  # header row
        for user, movie_list in recommendations.items():
            for movie, rank in movie_list:
                writer.writerow([user, movie, rank])
def convert_csv_to_tsv(csv_file_path, tsv_file_path):
    df = pd.read_csv(csv_file_path)  # Read the CSV file as a DataFrame using pandas
    df.to_csv(tsv_file_path, sep='\t', index=False)  # Convert and save the DataFrame as a TSV file using pandas


"""
    Cold Start Utils
"""
def get_cold_test_hetrec(dataset_path, test_path):
    # Load the dataset
    train = pd.read_csv(dataset_path, sep="\t", header=None, names=['userId','artistId','rating'])
    test = pd.read_csv(test_path, sep="\t", header=None, names=['userId','artistId','rating'])

    # Compute the number of ratings for each user
    user_ratings = train.groupby('userId').size().reset_index(name='num_ratings')

    # Compute the quartiles for the number of ratings
    quartiles = user_ratings['num_ratings'].quantile([0.25, 0.5, 0.75])

    # Retrieve the subset of cold users based on the quartiles
    cold_users = user_ratings[user_ratings['num_ratings'] <= quartiles[0.25]]

    return test[test["userId"].isin(cold_users['userId'].to_list())]

def get_cold_test(dataset_path, test_path):
    # Load the dataset
    data = pd.read_csv(dataset_path, sep="\t", header=None, names=["userId","movieId","rating"])
    test = pd.read_csv(test_path, sep="\t", header=None, names=['userId','movieId','rating','timestamp'])

    # Compute the number of ratings for each user
    user_ratings = data.groupby('userId').size().reset_index(name='num_ratings')

    # Compute the quartiles for the number of ratings
    quartiles = user_ratings['num_ratings'].quantile([0.25, 0.5, 0.75])

    # Retrieve the subset of cold users based on the quartiles
    cold_users = user_ratings[user_ratings['num_ratings'] <= quartiles[0.25]]

    return test[test["userId"].isin(cold_users['userId'].to_list())]

def get_cold_test_facebook(dataset_path, test_path):
    # Load the dataset
    train = pd.read_csv(dataset_path, sep="\t", header=None, names=['userId','bookId','rating'])
    test = pd.read_csv(test_path, sep="\t", header=None, names=['userId','bookId','rating'])

    # Compute the number of ratings for each user
    user_ratings = train.groupby('userId').size().reset_index(name='num_ratings')

    # Compute the quartiles for the number of ratings
    quartiles = user_ratings['num_ratings'].quantile([0.25, 0.5, 0.75])

    # Retrieve the subset of cold users based on the quartiles
    cold_users = user_ratings[user_ratings['num_ratings'] <= quartiles[0.25]]

    return test[test["userId"].isin(cold_users['userId'].to_list())]

class Utils:
    def __init__(self, data):
        self.data = data
    #                                   #
    # Counting token for Chat API Calls #
    #                                   #

    #                               #
    #   Utils for ChatGPT Tokens    #
    #                               #
    def num_tokens_from_string(self, model, message) -> int :
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(model) #gpt-3.5-turbo
        num_tokens = len(encoding.encode(message))
        return num_tokens
    def num_tokens_from_messages(self, model, messages):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":  # note: future models may deviate from this
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
      See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

    #                                       #
    #     Generate message for ChatGPT      #
    #                                       #
    """
        MovieLens Small
    """
    def movies_rated_by_user(self, user_id):
        user_movies = self.data[self.data['userId'] == user_id]
        movie_ratings = user_movies[['title', 'rating']]
        #movie_list = ', '.join(str(row['title']) + f" {row['rating']}/5.BK_0" for _, row in movie_ratings.iterrows())
        #movie_list = ', '.join(str(row['title']) + f" {int(row['rating'])}/5" for _, row in movie_ratings.iterrows())
        movie_list = ', '.join(str(row['title']) + f" {int(row['rating'])}/5" for _, row in movie_ratings.iterrows())
        #movie_list = ', '.join(str(row['title']) for _, row in movie_ratings.iterrows())
        sentence = f"You know that the user {user_id} likes the following movies: {movie_list}."
        #print(sentence)
        return sentence
    def rerank_by_user_profile(self, user_id):
        user_movies = self.data[self.data['userId'] == user_id]
        movie_ratings = user_movies[['title', 'rating']]
        movie_list = ', '.join(str(row['title']) + f" {int(row['rating'])}/5" for _, row in movie_ratings.iterrows())
        # The sentence have 467 tokens without considering the {user_id} and {movie_list}
        # The list of movies to re-rank are movies equally distributed by genres and popularity
        sentence = f"You know that user {user_id} likes the following movies: {movie_list}. " \
                   f"Re-rank the following movies into a top-50 recommendations: " \
                   f"Toy Story (1995), Lord of the Rings: The Fellowship of the Ring, The (2001), " \
                   f"Memento (2000), Star Wars: Episode IV - A New Hope (1977), Matrix, The (1999), " \
                   f"Fight Club (1999), Schindler's List (1993), Super Size Me (2004), Roger & Me (1989), " \
                   f"Maltese Falcon, The (1941), Apollo 13 (1995), Wizard of Oz, The (1939), Inception (2010), " \
                   f"Twelve Monkeys (a.k.a. 12 Monkeys) (1995), Bowling for Columbine (2002), Pulp Fiction (1994), " \
                   f"Tombstone (1993), American Beauty (1999), Star Wars: Episode V - The Empire Strikes Back (1980), " \
                   f"Princess Bride, The (1987), Fahrenheit 9/11 (2004), Seven (a.k.a. Se7en) (1995), Aliens (1986), " \
                   f"Willy Wonka & the Chocolate Factory (1971), Usual Suspects, The (1995), Good, the Bad and the Ugly, " \
                   f"The (Buono, il brutto, il cattivo, Il) (1966), Shawshank Redemption, The (1994), Alien (1979), " \
                   f"Lord of the Rings: The Two Towers, The (2002), Dark Knight, The (2008), Sixth Sense, The (1999), " \
                   f"Sin City (2005), Aladdin (1992), Terminator 2: Judgment Day (1991), Dances with Wolves (1990), " \
                   f"Silence of the Lambs, The (1991), Shrek (2001), Beauty and the Beast (1991), L.A. Confidential (1997)," \
                   f" Dark City (1998), Back to the Future Part III (1990), Jurassic Park (1993), Fargo (1996)," \
                   f" Forrest Gump (1994), Chinatown (1974), Finding Nemo (2003), Shining, The (1980), Lion King, The (1994)," \
                   f" Saving Private Ryan (1998), Braveheart (1995)"
        #print(sentence)
        return sentence
    def rerank_by_similar_user_profile(self, user_id, result):
        user_movies = self.data[self.data['userId'] == user_id]
        movie_ratings = user_movies[['title', 'rating']]
        movie_list = ', '.join(str(row['title']) + f" {int(row['rating'])}/5" for _, row in movie_ratings.iterrows())
        # The sentence have 467 tokens without considering the {user_id} and {movie_list}
        # The list of movies to re-rank are movies equally distributed by genres and popularity
        path = '../results/user_knn_models/user_knn_movielens_small_2018/' \
               'UserKNN_nn=80_sim=correlation_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=/' \
               'best-weights-UserKNN_nn=80_sim=correlation_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights='
        user_knn = UserKnnLoader(path)
        unseen_movie = search_50_unseen_movies(user_id, user_knn)
        for rank in range(1, 51):
            result.append((user_id, unseen_movie.reset_index()['movieId'][rank - 1], rank))
        unseen_movie_list = ', '.join(str(row['title']) for _, row in unseen_movie.iterrows())
        sentence = f"You know that user {user_id} likes the following movies: {movie_list}. " \
                   f"Re-rank the following movies into a top-50 recommendations: {unseen_movie_list}."
        #print(sentence)
        return sentence

    """
        HetRec2011
    """
    def artists_listened_by_user(self, user_id):
        user_artist = self.data[self.data['userId'] == user_id]
        artist_ratings = user_artist[['name', 'weight']].sort_values('weight', ascending=False)
        artist_list = ', '.join(str(row['name']) for _, row in artist_ratings.iterrows())
        sentence = f"You know that user {user_id} likes the following artists ordered by liking: {artist_list}."
        #print(sentence)
        return sentence
    def rerank_by_user_profile_hetrec(self, user_id):
        user_artist = self.data[self.data['userId'] == user_id]
        artist_ratings = user_artist[['name', 'weight']].sort_values('weight', ascending=False)
        artist_list = ', '.join(str(row['name']) for _, row in artist_ratings.iterrows())
        sentence = f"You know that user {user_id} likes the following artists ordered by liking: {artist_list}. " \
                   f"Re-rank the following artists into a top-50 recommendations: " \
                   f"Britney Spears, Depeche Mode, Lady Gaga, Christina Aguilera, Paramore, Madonna, " \
                   f"Rihanna, Shakira, The Beatles, Katy Perry, Avril Lavigne, Taylor Swift, Evanescence, " \
                   f"Glee Cast, Beyoncé, U2, 30 Seconds to Mars, Muse, Pink Floyd, Kylie Minogue, Miley Cyrus, " \
                   f"Radiohead, Ke$ha, Duran Duran, Coldplay, Blur, Arctic Monkeys, Placebo, The Killers, Nirvana, " \
                   f"Iron Maiden, Amy Winehouse, Led Zeppelin, Metallica, Michael Jackson, Pearl Jam, a-ha, Björk, " \
                   f"System of a Down, Daft Punk, Nine Inch Nails, Linkin Park, Kelly Clarkson, All Time Low, P!nk, " \
                   f"Avenged Sevenfold, Black Eyed Peas, My Chemical Romance, Green Day, The Cure."
        #print(sentence)
        return sentence
    def rerank_by_similar_user_profile_hetrec(self, user_id, result):
        user_artist = self.data[self.data['userId'] == user_id]
        artist_ratings = user_artist[['name', 'weight']].sort_values('weight', ascending=False)
        artist_list = ', '.join(str(row['name']) for _, row in artist_ratings.iterrows())
        path = '../results/user_knn_models/user_knn_hetrec2011_lastfm_2k/' \
               'UserKNN_nn=100_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=/' \
               'best-weights-UserKNN_nn=100_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights='
        user_knn = UserKnnLoader(path)
        unlisten_artist = search_50_unlisten_artist(user_id, user_knn)
        for rank in range(1, 51):
            result.append((user_id, unlisten_artist.reset_index()['id'][rank - 1], rank))
        unlisten_artist_list = ', '.join(str(row['name']) for _, row in unlisten_artist.iterrows())
        sentence = f"You know that user {user_id} likes the following artist: {artist_list}. " \
                   f"Re-rank the following artists into a top-50 recommendations: {unlisten_artist_list}."
        #print(sentence)
        return sentence

    """
    Facebook Book
    """
    def book_read_by_user(self, user_id):
        user_book = self.data[self.data['userId'] == user_id]
        book_ratings = user_book[['name', 'rating', 'bookId']].sort_values('rating', ascending=False)
        authors = get_author_name()
        book_list = ', '.join(str(row['name']) + f" by {authors[row['bookId']]}" for _, row in book_ratings.iterrows())
        sentence = f"You know that user {user_id} likes the following books: {book_list}."
        #print(sentence)
        return sentence

    def rerank_by_user_profile_facebook(self, user_id):
        user_book = self.data[self.data['userId'] == user_id]
        book_ratings = user_book[['name', 'rating', 'bookId']].sort_values('rating', ascending=False)
        authors = get_author_name()
        book_list = ', '.join(str(row['name']) + f" by {authors[row['bookId']]}" for _, row in book_ratings.iterrows())
        sentence = f"You know that user {user_id} likes the following books ordered by liking: {book_list}. " \
                   f">> Re-rank this list of books into a top-50 recommendations: " \
                   f"The Unbearable Lightness of Being, Dracula, The Catcher in the Rye, The Lost Books (novel series), " \
                   f"The Picture of Dorian Gray, The Great Gatsby, Brave New World, Harry Potter, The Pillowman, " \
                   f"Dragon Ball Z, Gantz, Twilight Eyes, The Maze Runner, Across the Universe (novel), The City of Ember, " \
                   f"Gravitys Rainbow, The Southern Vampire Mysteries, The Hobbit, Delta of Venus, Medea (play), Dexter in the Dark, " \
                   f"The Silmarillion, And the Ass Saw the Angel, Incarceron, Made in America (book), World War Z, " \
                   f"List of Scott Pilgrim characters, Frankenstein, W.I.T.C.H., If There Be Thorns, Petals on the Wind, " \
                   f"Cut (novel), Vampire Kisses, The Outsiders (novel), The Tenant of Wildfell Hall, Invisible Man, " \
                   f"Lady Chatterleys Lover, The Giver, Strange Case of Dr Jekyll and Mr Hyde, Crime and Punishment, " \
                   f"The Scarlet Letter, The Phantom of the Opera, A Series of Unfortunate Events, The Rapture (novel), " \
                   f"Shakespeares sonnets, The Lord of the Rings, The House of Mirth, Pygmalion (play), The Stand, " \
                   f"Through the Looking-Glass."
        #print(sentence)
        return sentence

    def rerank_by_similar_user_profile_facebook(self, user_id, result):
        user_book = self.data[self.data['userId'] == user_id]
        book_ratings = user_book[['name', 'rating', 'bookId']].sort_values('rating', ascending=False)
        authors = get_author_name()
        book_list = ', '.join(str(row['name']) + f" by {authors[row['bookId']]}" for _, row in book_ratings.iterrows())
        path = '../results/user_knn_models/user_knn_facebook_book/weights/' \
               'UserKNN_nn=60_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=/' \
               'best-weights-UserKNN_nn=60_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights='
        user_knn = UserKnnLoader(path)
        unread_book = search_50_unread_book(user_id, user_knn)
        for rank in range(1, 51):
            result.append((user_id, unread_book.reset_index()['bookId'][rank - 1], rank))
        unread_book_list = ', '.join(str(row['resourceURI']) for _, row in unread_book.iterrows())
        sentence = f"You know that user {user_id} likes the following books: {book_list}. " \
                   f">> Re-rank this list of books into a top-50 recommendations: {unread_book_list}."
        #print(sentence)
        return sentence

class SearchCache:
    def __init__(self):
        self.closest_matches_cache = {}
        self.hallucination_matches_cache = {}

    def clear_caches(self):
        self.closest_matches_cache.clear()
        self.hallucination_matches_cache.clear()