import tweepy
import csv
import json
import time

# Authenticate to Twitter
auth = tweepy.OAuthHandler("####", "####")
auth.set_access_token("####", "####")

# Create API object
api = tweepy.API(auth)


locations = {
	'Dar_es_Salaam': '-6.8,39.283333,100mi',
	'Dodoma': '-6.173056,35.741944,100mi'
}


def get_tweets(keyword,n,city=None, subfolder='general'):
	query = '%s -filter:retweets' % (keyword)
	if city is not None:
		query += "geocode:%s" % locations[city]
	tweets = tweepy.Cursor(api.search, q=query, tweet_mode="extended", wait_on_rate_limit=True).items(n)
	while True:
		tweet = tweets.next()
		with open("tweets/%s/%s.json" % (subfolder,tweet.id_str), "w") as outfile: 
			outfile.write(json.dumps(tweet._json,indent=4))

# get general tweets
get_tweets('Dar_es_Salaam','',1000000,'general')

# get covid-related tweets
get_tweets('(covid OR corona OR covid OR Korona)',10000,'Dar_es_Salaam',subfolder='covid')
get_tweets('(covid OR corona OR covid OR Korona)',10000,'Dodoma',subfolder='covid')
get_tweets('(tanzania OR Magufulli) (corona OR covid OR Korona)',10000,None,subfolder='covid')


# get Magufuli related tweets
get_tweets('(to:MagufuliJP) OR (@MagufuliJP)',1000000,None,subfolder='magufuli')
get_tweets('Magufuli|president',10000,'Dar_es_Salaam',subfolder='magufuli')
get_tweets('Magufuli|president',10000,'Dodoma',subfolder='magufuli')
