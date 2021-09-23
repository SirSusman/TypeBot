import requests,pickle,re,os,pandas as pd, string
from sklearn.feature_extraction.text import CountVectorizer

#/Users/brandon/opt/anaconda3/envs/type_bot/bin/python
"""first_ep = 25301
last_ep = 25498
pilot = "https://transcripts.foreverdreaming.org/viewtopic.php?f=574&t=25301&sid=67f5b896a0a47ed09f5f4d9eb3036fd9"
urls = []"""

#print(scene_data)
# Data Structure: ["name", "line"]
def buildCorpuses(scene_data):
	corpuses = {}
	for key in scene_data:
		for line in scene_data[key]:
			character = line[0]
			text = line[1]
			if(character in corpuses.keys()):
				#add line to the corpus
				#print("Character Found")
				corpuses[character] += text + ' '
			else:
				#Name not in dictionary update it
				#print("Who??")
				corpuses[character] = text + ' '
			
			#print(f"**Character**{character}, **Text**{text}")
	return corpuses


#We want to save corpuses as character_name_corpus.pkl

def cleaningRound1(text):
	text = text.lower()
	#gets rid of these literal characters
	text = re.sub('[.*?,\']', '', text)
	#gets rid of apostrophes
	text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
	#gets rid of digits
	text = re.sub('\w*\d\w*', '', text)
	return text
	# Apply a second round of cleaning
def cleaningRound2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…[]]', '', text)
    text = re.sub('\n', '', text)
    return text


		

#Save corpuses
def pickleData(data_frame,name):
	with open(os.getcwd()+"/corpuses/" + name + ".pkl","wb") as file:
			pickle.dump(data_frame,file)
def loadPickle(input_file):
#Load corpuses
	with open(os.getcwd()+'/corpuses/' + str(input_file) + '.pkl',"rb") as file:
			return pickle.load(file)




def recreateData(cleaning1,cleaning2,input_dict):
	print(input_dict)
	#Convert character corpuses into data frames
	corpus_df = pd.DataFrame.from_dict(input_dict).transpose()
	corpus_df.columns = ['transcript']
	corpus_df = corpus_df.sort_index()
	print(corpus_df.index)


	
	#More cleaning could be done like stemming, bi-grams (thank you becomes thankyou).
	#df_clean = pd.DataFrame(corpus_df.transcript.apply(cleaning1))
	#df_clean = pd.DataFrame(df_clean.transcript.apply(cleaning2))
	
	#pickleData(corpus_df,"corpus_dict")
	#pickleData(df_clean,"df_clean")
	#return corpus_df,df_clean

def loadData(file1,file2):
	corpus_df = loadPickle(file1)
	df_clean = loadPickle(file2)
	return corpus_df, df_clean


def tokenizeData(df_clean):
	#Tokenizes words
	cv = CountVectorizer(stop_words = 'english')
	#Provide the vocabulary by giving it the cleaned corpuses
	data_cv = cv.fit_transform(df_clean.transcript)
	#Gets all the words you've provided it
	#Document Term Matrix
	#the data for the Document term matrix is the array format of the cv, and the columns are
	#the vocab you gave it.
	data_dtm = pd.DataFrame(data_cv.toarray(),columns =cv.get_feature_names())
	#Sort alphabetically
	data_dtm.index = df_clean.index
	pickleData(data_dtm,"data_dtm")

def main():
	#Lambda takes 'x' and does cleaingRound1(x) <-- this with it
	#Data Frame apply automatically passes the stuff into the lambda
	round1 = lambda x: cleaningRound1(x)
	round2 = lambda x: cleaningRound2(x)
	character_dict = pickle.load(open("character_dict.p","rb"))

	for key in character_dict:
		character_dict[key] = [" ".join(value for value in character_dict[key])]

	#print(character_dict)
	recreateData(round1,round2,character_dict)
	#print(character_dict)
	#Now we have the character dict as a list of strings.
	# We have to consolidate into one string, then go through the cleaning rounds and convert it
	# to a dataframe. You can use " ".join



	#recreateData(round1, round2)
	corpus_df, df_clean = loadData("corpus_dict","df_clean")
	#print(corpus_df.transcript.loc['Michael'])
	#print(df_clean.transcript.loc['Michael'])
	
	#pass in data frame
	tokenizeData(df_clean)
	#data_dtm = loadPickle("data_dtm")

	#dtm = pd.read_pickle("corpuses/dtm.pkl")
	#data_dtm = loadPickle("dtm")
	#data_dtm = data_dtm.transpose()
	#print(data_dtm)
	#pickleData(data_dtm,"dtm")
	#print(data_dtm.loc[['Angela',"Michael","Mr. Brown"]])
	#print(data_dtm)
	"""
	top30_dict = {}
	for c in data_dtm.columns:
		top = data_dtm[c].sort_values(ascending = False).head(30)
		top30_dict[c] = list(zip(top.index,top.values))

	print(top30_dict)
	for key,value in top30_dict.items():
		print(key)
		for v in value:
			print(v)
	
	print("Done")
"""

#=============Functions===============
main()
