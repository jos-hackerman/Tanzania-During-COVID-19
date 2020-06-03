import polyglot
from polyglot.text import Text, Word
import glob 
import os
from shutil import copyfile
import pandas as pd
import tqdm
import numpy as np
import re
import json
from datetime import datetime
import nltk
import tqdm
import multiprocessing as mp
nltk.download('movie_reviews')
nltk.download('punkt')
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

from pandarallel import pandarallel
pandarallel.initialize()

##################################################
# helper functions
##################################################

def get_score(row):
	text = Text(row.full_text)
	try: 
		score = sum([word.polarity for word in text.words])
	except ValueError as e:
		score = np.NaN
	return score


def lang_detector(texts):
	text = Text(texts)
	return text.language.code

def clean_texts(texts):
	# remove @
	texts = re.sub('@\w+\W','',texts)
	# remove urls from text
	return ' '.join(re.sub("https.*", "", texts).split())

def extract_info(tweet,i):
  print(i)
  blob = TextBlob(tweet, analyzer=NaiveBayesAnalyzer())
  cat, pos, neg = blob.sentiment
  return {'cat':cat,'pos':pos,'neg':neg}


########################################
# copy file to the same location
########################################

for folder in ['tweets/covid','tweets/magufuli','tweets/general']:
	all_files = glob.glob('%s/*.json' % folder)
	for fname in tqdm.tqdm(all_files):
		fname_dest = os.path.join('tweets/all/',os.path.basename(fname))
		if os.path.exists(fname_dest):
			continue
		copyfile(fname,fname_dest)

##################################################
# read twitter data 
##################################################

dict_tweet = []
for fname in tqdm.tqdm(glob.glob('tweets/all/*.json')):
	with open(fname) as json_file:
		data = json.load(json_file)
		dict_inst_tweet = {}
		dict_inst_tweet['id_str']                    = data['id_str']
		dict_inst_tweet['created_at']                = datetime.strptime(data['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
		dict_inst_tweet['full_text']                 = data['full_text']
		dict_inst_tweet['in_reply_to_status_id_str'] = data['in_reply_to_status_id_str']
		dict_inst_tweet['in_reply_to_user_id_str']   = data['in_reply_to_user_id_str']
		dict_inst_tweet['retweet_count']             = data['retweet_count']
		dict_inst_tweet['favorite_count']            = data['favorite_count']
		dict_inst_tweet['lang']                      = data['lang']
		dict_inst_tweet['user_id']                   = data['user']['id']
		dict_inst_tweet['location']                  = data['user']['location']
		dict_tweet.append(dict_inst_tweet)

df = pd.DataFrame(dict_tweet)
df['general'] = 0
df.loc[df.id_str.isin([x.split('.')[0] for x in glob.glob('tweets/general/*.json')]),'general'] = 1
df.to_pickle('tweets/tweets_raw.pkl')

##################################################
# language detection
##################################################
df['full_text_raw'] = df['full_text']
df['full_text'] = df['full_text_raw'].apply(clean_texts)
df['lang_detected'] = df.full_text.parallel_apply(lang_detector)
df.lang_detected.value_counts()



##################################################
# translate swahili tweets
##################################################

from googletrans import Translator
translator = Translator()

df_sw = df.loc[df.lang_detected == 'sw'].copy()
df_sw['full_text_cleaned'] = df_sw.full_text.str.replace('[^\w\s]|_','')
df_sw = df_sw.reset_index(drop=True)
df_sw['translated'] = 0
df_sw['full_text_trans'] = ''

# because google's limit, I will need to change network address periodically
# I am using purevpn to do this
# locations of purevpn
# https://support.purevpn.com/article-categories/getting-started/linux
locations = np.array([
	"AF","AL","DZ","AO","AR","AM","AW","AU","AT","AZ","BH","BD","BB","BE","BZ","BM",
	"BO","BA","BR","VG","BN","BG","KH","CA","CV","KY","CL","CN","CO","CR","HR","CY",
	"CZ","DK","DM","DO","EC","EG","SV","EE","ET","FI","FR","GE","DE","GH","GR","GD",
	"GT","GY","HT","HN","HK","HU","IS","IN","ID","IE","IM","IT","JM","JP","JO","KZ",
	"KE","KR","KW","KG","LA","LV","LB","LI","LT","LU","MO","MG","MY","MT","MR","MU",
	"MX","MD","MC","MN","ME","MS","MA","mm","NL","NZ","NI","NE","NG","NO","OM","PK",
	"PA","PG","PY","PE","PH","PL","PT","PR","QA","RO","RU","LC","SA","SN","RS","SC",
	"SG","SK","SI","ZA","ES","LK","SR","SE","CH","SY","TW","TJ","TZ","TH","BS","TT",
	"TN","TR","TM","TC","UA","AE","GB","US","UY","UZ","VE","VN","YE",])

texts = []
translated = []
indices_translating = []
for i,row in df_sw.loc[df_sw.translated == 0].iterrows():
	if (sum([len(x) for x in texts]) + len(row.full_text_cleaned) > 15000) or (i == (df_sw.shape[0]-1)):
		if i == (df_sw.shape[0]-1):
			texts.append(row.full_text_cleaned.replace('\n',' ').encode('ascii', 'ignore').decode('ascii'))
			indices_translating.append(i)
		while True:
			try:
				print("Sending to translation: %s" % i)
				trans = translator.translate(texts)
				break
			except:
				# disconnect vpn
				os.system('purevpn -d')
				# re-connect vpn with new location
				os.system('purevpn -c %s' % np.random.choice(locations,1)[0])
				translator = Translator()
		for tran in trans:
			translated.append(tran.text)
		df_sw.iloc[indices_translating,df_sw.columns.to_list().index('full_text_trans')] = translated
		df_sw.iloc[indices_translating,df_sw.columns.to_list().index('translated')] = 1
		# backup the work
		df_sw.to_pickle('tweets/df_sw_translating.pkl')
		translated = []
		indices_translating = []
		texts = []

	texts.append(row.full_text_cleaned.replace('\n',' ').encode('ascii', 'ignore').decode('ascii'))
	indices_translating.append(i)

# add back to the original dataset
df = df.merge(df_sw[['id_str','full_text_trans','translated']],on='id_str',how='left')

df['tweet'] = df.full_text
df.loc[df.translated == 1, 'tweet'] = df.loc[df.translated == 1, 'full_text_trans']

##################################################
# extract sentiment scores
##################################################
masks = df.lang_detected.isin(['sw','en'])
print("Parallel using %s cpus" % mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
results_raw = pool.starmap(extract_info, [(tweet,i) for i,tweet in enumerate(df.loc[masks].tweet.to_list())])
pool.close()
df.loc[masks] = results_raw

df.to_pickle('tweets/tweets_all.pkl')

##################################################
# get twitter users 
##################################################

df['user_id_str'] = df.user_id.apply(str)
all_users = df.user_id.apply(str).to_list() + df.in_reply_to_user_id_str.to_list()

df_user = pd.DataFrame(all_users)
df_user.columns = ['user_id_str']
df_user = df_user.drop_duplicates()
df_user = df_user.loc[~df_user.user_id_str.isna()]

df_user = df_user.merge(df[['user_id_str','location','lang_detected']].drop_duplicates('user_id_str'),on='user_id_str',how='left')

import tweepy
import csv
import json
import time

# Authenticate to Twitter
auth = tweepy.OAuthHandler("####", "###")
auth.set_access_token("####", "####")

# Create API object
api = tweepy.API(auth)
for i,row in df_user.iterrows():
	if pd.isnull(row.location) and not os.path.exists("tweets/users/%s.json" % (row.user_id_str)):
		try:
			print("Finding user %s ..........." % row.user_id_str)
			tu = api.get_user(row.user_id_str)
			print("User %s found!" % row.user_id_str)
			with open("tweets/users/%s.json" % (tu.id_str), "w") as outfile: 
				outfile.write(json.dumps(tu._json,indent=4))
		except tweepy.TweepError as e:
			print(e)
			if e.args[0][0]['code'] in [50, 63]:
				pass
			else:
				time.sleep(60)
		except:
			pass
	else:
		print("User %s already exist!" % row.user_id_str)


dict_user = []
for fname in glob.glob('tweets/users/*.json'):
	with open(fname) as json_file:
		data = json.load(json_file)
		dict_inst_tweet = {}
		dict_inst_tweet['user_id_str']  = data['id_str']
		dict_inst_tweet['location']     = data['location']
		dict_user.append(dict_inst_tweet)

df_user_new = pd.DataFrame(dict_user)
df_user = pd.concat([df_user_new,df_user.loc[~pd.isnull(df_user.location)]])
df_user.to_pickle('tweets/users_raw.pkl')


##################################################
# normalize locations of users
##################################################
def clean_location(location):
	location = re.sub('[^a-zA-Z]',' ', location)
	return re.sub(' +', ' ', location)

df_user['location'] = df_user['location'].str.lower().str.strip()
df_user['location'] = df_user.location.apply(clean_location)

df_user.loc[df_user.location.isin(['n/a','',' ']),'location'] = 'unknown'
df_user.loc[df_user.location.isin(['nairobi','nairobi - kenya','nairobi,kenya','nairobi kenya','mombasa, kenya','nairobi, kenya','nairobi','kisumu, kenya','eldoret, kenya','uganda, kenya','nakuru, kenya','nairobi, kenya.','watamu, kenya']),'location'] = 'kenya'
df_user.loc[df_user.location.isin(['tanzania dar es salaam', 'pugu dar es salaam','tanzania,dar esalaam tabata.','dar es salaam - tanzania','88.5 fm dar es salaam','dar,tanzania','dsm','temeke,dar es salaam,tanzania','dar es salaam, tanzania🇹🇿','dar-es-salaam,tanzania','dar es salaam', 'dar es salaam tanzania','dar-es-salaam - tanzania','dar es salaam','dar-es-salaam, tanzania','dar es salaam,tanzania']),'location'] = 'dar es salaam tanzania'
df_user.loc[df_user.location.isin(['kampala, uganda','uganda','kampala']),'location'] = 'uganda'
df_user.loc[df_user.location.isin(['pretoria, south africa',"mbede, nigeria","kigali, rwanda",'cape town, south africa','morogoro','rwanda','africa.','accra, ghana','lagos, nigeria','middle of africa','south africa','africa','johannesburg, south africa', 'east africa', 'nigeria','lusaka, zambia']),'location'] = 'africa'
df_user.loc[df_user.location.isin(['shinyanga, tanzania','tanzania 🇹🇿','dodoma tanzania','tanzania, kenya and uganda','dodoma-tanzania','arusha tanzania','zanzibar, tanzania','lusaka zambia ,dar  tanzania','#dodoma🏙#tanzania','pemba north, tanzania','tanzania & zanzibar','dodoma,tanzania','arusha,tanzani','mtwara, tanzania','singida, tanzania','njombe','dodoma','geita vijijini','arusha','mondo','mara, tanzania','pwani, tanzania','kagera, tanzania','mwandiga township, kigoma','kigoma, tanzania','njombe,tanzania','tanzania🇹🇿','zanzibar south and central, ta','katende, chato tanzania','tanga, tanzania','tanzania / uk','kilimanjaro, tanzania','mbeya, tanzania','huku area tu','morogoro, tanzania','dodoma, tanzania','zanzibar','zanzibar west, tanzania', 'iringa, tanzania', 'mwanza, tanzania','tanzania','westlands','tanga/tanzania','arusha, tanzania']),'location'] = 'other tanzania'
df_user.loc[df_user.location.isin(['los angeles, ca','new york, usa','bogotá, d.c., colombia','new york, ny','united states','bronx, ny','washington, dc','brooklyn, new york','usa','denver','texas, usa']),'location'] = 'us'
df_user.loc[df_user.location.isin(['west midlands, england','london','scotland','london, england','united kingdom']),'location'] = 'uk'
df_user.loc[df_user.location.isin(['global','oh and putin killed epstein','respecthumanrights','global citizen','planet earth, last time i chkd','worldwide','earth','australia','dubai, united arab emirates','berlin, germany','united arab emirates','berlin, germany','united arab emirates','moscow, russia','méxico','india','toronto, ontario','europe','baghdad-iraq']),'location'] = 'other'

df_user.loc[df_user.location.str.contains('uganda'),'location'] = 'uganda'
df_user.loc[df_user.location.str.contains('angeles|washington|ny|usa|atlanta'),'location'] = 'us'
df_user.loc[df_user.location.str.contains('england'),'location'] = 'uk'
df_user.loc[df_user.location.str.contains('salaam'),'location'] = 'dar es salaam tanzania'
df_user.loc[df_user.location.str.contains('dar'),'location'] = 'dar es salaam tanzania'
df_user.loc[(df_user.location.str.contains('tanzania')) & (~ df_user.location.str.contains('salaam')),'location'] = 'other tanzania'
df_user.loc[(df_user.location.str.contains('tz')) & (~ df_user.location.str.contains('salaam')),'location'] = 'other tanzania'
df_user.loc[df_user.location.str.contains('dar'),'location'].value_counts()
df_user.location.value_counts().head(30)

df_user['location_cat'] = 'Tanzania'
df_user.loc[df_user.location.isin(['kenya', 'uganda', 'africa']),'location_cat'] = 'Africa'
df_user.loc[df_user.location.isin(['us', 'uk', 'other']),'location_cat'] = 'Other'
df_user.loc[df_user.location.isin(['unknown']),'location_cat'] = 'Unknown'
df_user.location_cat.value_counts()
df_user.to_pickle('tweets/users.pkl')

##################################################
# clean topics 
##################################################
df = df.drop(['location','location_cat'],axis=1).merge(df_user[['user_id_str','location','location_cat']],on='user_id_str',how='left')

df['topic_reply_magufuli'] = (df.in_reply_to_user_id_str == '3378377920').astype(int)
df['topic_magufuli'] = (df.tweet.str.lower().str.contains('magufuli|president', regex=True).astype(int) + df['topic_reply_magufuli']).gt(0).astype(int)
df['topic_tanz'] = df.tweet.str.lower().str.find('tanz').gt(0).astype(int)
df['topic_covid'] = df.tweet.str.lower().str.contains('covid|corona|korona', regex=True).astype(int)
df['topic_in_tanz'] = (df.location.str.lower().str.find('tanz').gt(0).astype(int) + df.topic_tanz + df.topic_magufuli).gt(0).astype(int)
df['topic_covid_in_tanz'] = (df.topic_covid * df.topic_in_tanz).gt(0).astype(int)

df.topic_covid.describe()
df.topic_magufuli.describe()
df.topic_in_tanz.describe()

df['location_in_tanz'] = (df.location.str.lower().str.find('tanz').gt(0).astype(int) + df.location_cat.str.lower().str.find('tanz').gt(0).astype(int)).gt(0).astype(int)
df.loc[df.general == 1, 'location_in_tanz'] = 1

df.to_pickle('tweets/tweets_all_cleaned.pkl')