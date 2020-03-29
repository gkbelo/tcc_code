import pandas as pd
from sklearn.model_selection import train_test_split

def read_artists():
    cols = ['artist_id', 'artist_mbid', 'artist_name']
    df = pd.read_csv('mmtd/artists.txt', sep='\t', names=cols)
    df = df.drop(['artist_mbid'], axis=1)
    return df


def read_tracks():
    cols = ['track_id', 'track_title', 'track_artistId']
    df = pd.read_csv('mmtd/track.txt', sep='\t', names=cols)
    return df


def read_tweets():
    cols = ['tweet_id', 'tweet_tweetId', 'tweet_userId', 'tweet_artistId', 'tweet_trackId', 'tweet_datetime',
            'tweet_weekday', 'tweet_longitude', 'tweet_latitude']
    df = pd.read_csv('mmtd/tweet.txt', sep='\t', names=cols)
    df = df.drop(['tweet_datetime', 'tweet_weekday', 'tweet_longitude', 'tweet_latitude'], axis=1)
    df['tweet_count'] = 1
    return df


class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, tweets_df, tracks_df, artists_df):
        self.tweets_df = tweets_df
        self.tracks_df = tracks_df
        self.artists_df = artists_df

    def get_model_name(self):
        return self.MODEL_NAME

    def top_songs(self, topn=50):
        # Create the dataset with the most popular songs
        top_df = self.tweets_df.groupby('tweet_trackId')['tweet_count'].sum().sort_values(ascending=False).reset_index()
        top_df = top_df.rename(columns={"tweet_trackId": "track_id"})
        top_df = pd.merge(top_df, self.tracks_df, left_on='track_id', right_on='track_id', how='left')
        top_df = pd.merge(top_df, self.artists_df, left_on='track_artistId', right_on='artist_id', how='left')
        top_df = top_df.drop(['artist_id', 'track_artistId'], axis=1)
        return top_df.head(topn)

    def recommend_items(self, items_to_ignore=[], topn=10):
        # Recommend the more popular items that the user hasn't seen yet.
        pop_df = self.top_songs(topn)
        recom_df = pop_df[~pop_df['track_id'].isin(items_to_ignore)] \
            .sort_values('tweet_count', ascending=False)
        return recom_df.head(topn)


# read the dataset files
# Artists file
artists = read_artists()
#print(artists.head())

# Tracks file
tracks = read_tracks()
#print(tracks.head())

# Tweets file
tweets = read_tweets()
#print(tweets.head())

# Split data between train and test dataset
tweet_train, tweet_test = train_test_split(tweets, test_size=0.2)
print('Size of TRAIN dataset: ' + str(tweet_train.shape))
print('Size of TEST dataset: ' + str(tweet_test.shape))

# Popularity model - simple analysis
popularity_model = PopularityRecommender(tweets, tracks, artists)

print('Top 5 songs')
print(popularity_model.top_songs(5))
print('Popular song recommender')
print(popularity_model.recommend_items(['141574'], 5))

# collaborative filtering model
