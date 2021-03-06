import pandas as pd

!curl --remote-name \
     -H 'Accept: application/vnd.github.v3.raw' \
     --location https://raw.githubusercontent.com/gkbelo/tcc_code/master/mmtd/artists.txt

!curl --remote-name \
     -H 'Accept: application/vnd.github.v3.raw' \
     --location https://raw.githubusercontent.com/gkbelo/tcc_code/master/mmtd/track.txt

!curl --remote-name \
     -H 'Accept: application/vnd.github.v3.raw' \
     --location https://raw.githubusercontent.com/gkbelo/tcc_code/master/mmtd/tweet.txt

def read_artists():
    cols = ['artist_id', 'artist_mbid', 'artist_name']
    df = pd.read_csv('artists.txt', sep='\t', names=cols)
    df = df.drop(['artist_mbid'], axis=1)
    return df

def read_tracks():
    cols = ['track_id', 'track_title', 'track_artistId']
    df = pd.read_csv('track.txt', sep='\t', names=cols)
    return df

def read_tweets():
    cols = ['tweet_id', 'tweet_tweetId', 'tweet_userId', 'tweet_artistId', 'tweet_trackId', 'tweet_datetime',
            'tweet_weekday', 'tweet_longitude', 'tweet_latitude']
    df = pd.read_csv('tweet.txt', sep='\t', names=cols)
    df = df.drop(['tweet_weekday', 'tweet_longitude', 'tweet_latitude', 'tweet_datetime'], axis=1)
    df['tweet_count'] = 1
    return df

def get_all_tracks_by_user(tweet_df, user_id):
    ids = tweet_df.drop(['tweet_id', 'tweet_tweetId', 'tweet_artistId', 'tweet_count'], axis=1)
    ids = ids.loc[ids['tweet_userId'] == user_id]
    ids.drop_duplicates(subset ="tweet_trackId", 
                        keep = False,
                        inplace = True)
    return ids['tweet_trackId'].tolist()

def get_all_users_by_track(tweet_df, track_id):
    ids = tweet_df.drop(['tweet_id', 'tweet_tweetId', 'tweet_artistId', 'tweet_count'], axis=1)
    ids = ids.loc[ids['tweet_trackId'] == track_id]
    ids.drop_duplicates(subset ="tweet_userId",
                        keep = False,
                        inplace = True)
    return ids['tweet_userId'].tolist()

class pop_songs:
    def __init__(self, tweets_df, tracks_df, artists_df):
        self.tweets_df = tweets_df
        self.tracks_df = tracks_df
        self.artists_df = artists_df


    def top_songs(self, topn=10):
        # Create the dataset with the most popular songs
        top_df = self.tweets_df.groupby('tweet_trackId')['tweet_count'].sum().sort_values(ascending=False).reset_index()
        top_df = top_df.rename(columns={"tweet_trackId": "track_id"})
        top_df = pd.merge(top_df, self.tracks_df, left_on='track_id', right_on='track_id', how='left')
        top_df = pd.merge(top_df, self.artists_df, left_on='track_artistId', right_on='artist_id', how='left')
        top_df = top_df.drop(['artist_id', 'track_artistId', 'tweet_count', 'track_id'], axis=1)
        return top_df.head(topn)


    def print_top5(self):
        print('-- Top 5 songs --')
        print(self.top_songs(5))
        print('--')

class similar_songs:
    def __init__(self, tweets_df, tracks_df, artists_df, user_tracks):
        self.tweets_df = tweets_df
        self.tracks_df = tracks_df
        self.artists_df = artists_df        
        self.user_tracks = user_tracks


    def recommend(self, ignore_tracks=[], topn=10, ignore_user_id=0):
      users_all = self.tweets_df.drop(['tweet_id', 'tweet_tweetId', 'tweet_artistId'], axis=1)
      users_filtered_df = pd.DataFrame()

      for idx_t in range(len(self.user_tracks)):
        # get all users that tweeted the same songs that the target user
        users_id_by_track = get_all_users_by_track(tweets, self.user_tracks[idx_t])
        if ignore_user_id > 0:
          users_id_by_track.remove(ignore_user_id)

        for idx_u in range(len(users_id_by_track)):
          # get all the tracks for each user
          tmp = users_all.loc[users_all['tweet_userId'] == users_id_by_track[idx_u]]
          users_filtered_df = pd.concat([tmp, users_filtered_df], ignore_index=True)

      recom_df = users_filtered_df[~users_filtered_df['tweet_trackId'].isin(ignore_tracks)] \
                  .sort_values('tweet_count', ascending=False)
      recom_df = recom_df.groupby('tweet_trackId')['tweet_count'].sum().sort_values(ascending=False).reset_index()
      recom_df = recom_df.rename(columns={"tweet_trackId": "track_id"})
      recom_df = pd.merge(recom_df, self.tracks_df, left_on='track_id', right_on='track_id', how='left')
      recom_df = pd.merge(recom_df, self.artists_df, left_on='track_artistId', right_on='artist_id', how='left')
      recom_df = recom_df.drop(['artist_id', 'track_artistId', 'tweet_count', 'track_id'], axis=1)
      return recom_df.head(topn)

"""Read the dataset files"""

artists = read_artists()
tracks = read_tracks()
tweets = read_tweets()

"""Exploring the dataset"""

print('Tweets numbers')
print('# lines: ' + str(tweets.shape[0]))

total_users = read_tweets()
total_users.drop_duplicates(subset ="tweet_userId", 
                            keep = False,
                            inplace = True)
print('Total of unique users: ' + str(total_users.shape[0]))

total_tracks = read_tweets()
total_tracks.drop_duplicates(subset ="tweet_trackId", 
                             keep = False,
                             inplace = True)
print('Total of unique songs: ' + str(total_tracks.shape[0]))

"""** Popularity model **    
Simple analysis with Top 5 songs
"""

pop_model = pop_songs(tweets, tracks, artists)
pop_model.print_top5()

"""Get the user id to start the process"""

# example users
#   265101134  (14 tweets)
#   58937384	 (854 tweets)
#   92235951   (75 tweets)
#   250253081	 (2 tweets)
#
# new user
#   43254
#
test_user = 58937384
#
tot_user_rows = read_tweets()
tot_user_rows = tot_user_rows.loc[tot_user_rows['tweet_userId'] == test_user]
print('-- User tweets count --')
print(str(tot_user_rows.shape[0]))

"""** Recommendation **  
Check the data for the target user
"""

# All songs tweeted by the target user
user_tracks = get_all_tracks_by_user(tweets, test_user)

# It's a new user then recommend the Top Songs
if len(user_tracks) == 0:
  print('-- Popular song recommendation --')
  print(pop_model.top_songs(topn=10))
else:
# If not a new user then recommend similar songs
  sim_song = similar_songs(tweets, tracks, artists, user_tracks)
  print('-- Song recommendation --')
  print(sim_song.recommend(ignore_tracks=user_tracks, ignore_user_id=test_user))