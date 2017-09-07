# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

#
# Note: This script was originaly contained inside a ipython notebook. 
#       Therefore the formating might seems a little weird without the cells format.  
#

# <codecell>

import nltk
import scipy
from sklearn import linear_model
import numpy
from sklearn.naive_bayes import MultinomialNB

# <codecell>

nrcPosDict={}
nrcNegDict={}
emoticonPosDict={}
emoticonNegDict={}
hashtagPosDict={}
hashtagNegDict={}

emoticonScoreDict={}
hashtagScoreDict={}

#Load lexicons into dictionaries
def makeDicts():
    datafile=file("lexicons/formated/nrc-emotion-Pos-FINAL.txt")
    for line in datafile:
        tokens = nltk.word_tokenize(line)
        for token in tokens:
            nrcPosDict[token]=1
    datafile=file("lexicons/formated/nrc-emotion-Neg-FINAL.txt")
    for line in datafile:
        tokens = nltk.word_tokenize(line)
        for token in tokens:
            nrcNegDict[token]=1
            
    datafile=file("lexicons/formated/emoticonPosFINAL.txt")
    for line in datafile:
        tokens = nltk.word_tokenize(line)
        for token in tokens:
            emoticonPosDict[token]=1
    datafile=file("lexicons/formated/emoticonNegFINAL.txt")
    for line in datafile:
        tokens = nltk.word_tokenize(line)
        for token in tokens:
            emoticonNegDict[token]=1

    datafile=file("lexicons/formated/hashTagPosFINAL.txt")
    for line in datafile:
        tokens = nltk.word_tokenize(line)
        for token in tokens:
            hashtagPosDict[token]=1
    datafile=file("lexicons/formated/hashTagNegFINAL.txt")
    for line in datafile:
        tokens = nltk.word_tokenize(line)
        for token in tokens:
            hashtagNegDict[token]=1
    
    datafile=file("lexicons/formated/EmoticonFINALScored.txt")
    for line in datafile:
        splitLine = line.split()
        emoticonScoreDict[splitLine[0]] =float(splitLine[1])
        
    datafile=file("lexicons/formated/HashtagFINALScored.txt")
    for line in datafile:
        splitLine = line.split()
        hashtagScoreDict[splitLine[0]] =float(splitLine[1])

# <codecell>

import string
posNRC = "lexicons/formated/nrc-emotion-Pos-FINAL.txt"
negNRC = "lexicons/formated/nrc-emotion-Neg-FINAL.txt"

#Compute features given a message (tweet,phrases,sms,etc)
def twitter_features(message):
    features={}
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    tokens = nltk.word_tokenize(message)
    nbOfCapitalizedWords=0
    posSmileys=[":)","=D",";)"]
    hasPosSmileys=False
    negSmileys=[":(","=(",";("]
    hasNegSmileys=False
    
    profanity=["shit","fuck","bitch","dick","cunt","asshole"]
    hasProfanity=False
    
    negationWords=["not","never","no","none","nobody"]
    suffixToAdd=""
    
    #When a negation words is detected all following word will be suffixed to indicate possible polarity inversion.
    for i in xrange(len(tokens)):
        
        if tokens[i] in negationWords:
            suffixToAdd="_neg"
        else:
            tokens[i]=tokens[i]+suffixToAdd
        
    
    upperCaseLetters = [l for l in message if l.isupper()]
      
    nbPosNRC = 0
    nbNegNRC=0
    
    nbPosEmo=0
    nbNegEmo=0
    
    nbPosHash=0
    nbNegHash=0
    
    nrcEmoScore=0
    nrcHashScore=0
    
    for token in tokens:
        features['contains(%s)' % token] = (token in tokens)
        if token.isupper():
            nbOfCapitalizedWords=nbOfCapitalizedWords+1
        if token in nrcPosDict:
            nbPosNRC=nbPosNRC+1
        if token in nrcNegDict:
            nbNegNRC=nbNegNRC+1
            
        if token in emoticonPosDict:
            nbPosEmo=nbPosEmo+1
        if token in emoticonNegDict:
            nbNegEmo=nbNegEmo+1
            
        if token in hashtagPosDict:
            nbPosHash=nbPosHash+1
        if token in hashtagNegDict:
            nbNegHash=nbNegHash+1
        
        if token[-4:]=="_neg":
            if token[:-4] in emoticonScoreDict:
                nrcEmoScore=nrcEmoScore- emoticonScoreDict[token[:-4]]

            if token[:-4] in hashtagScoreDict:
                nrcHashScore=nrcHashScore- hashtagScoreDict[token[:-4]]

        else:    

            if token in emoticonScoreDict:
                nrcEmoScore=nrcEmoScore+ emoticonScoreDict[token]

            if token in hashtagScoreDict:
                nrcHashScore=nrcHashScore+ hashtagScoreDict[token]

            
    for smileys in posSmileys:
        if smileys in message:
            hasPosSmileys=True
    for smileys in negSmileys:
      
        if smileys in message:
            hasNegSmileys=True
    for smileys in profanity:
        if smileys in message:
            hasProfanity=True
    
    
    
    nrcScore=nbPosNRC-nbNegNRC
    nrcDecision=nrcScore>=0
    nrcEmoDecision=nrcEmoScore >=0
    nrcHashDecision=nrcHashScore >=0
    globalLexiconScore=nrcScore+nrcEmoScore+nrcHashScore
    globalLexiconDecision=globalLexiconScore>=0
    
    
    nb_punct = count(message, string.punctuation)
    nb_exclamation= count(message, "!")
    
    features["nbNRCPos"]=nbPosNRC
    features["nbNRCNeg"]=nbNegNRC
    features["nbEmoPos"]=nbPosEmo
    features["nbEmoNeg"]=nbNegEmo
    features["nbHashPos"]=nbPosHash
    features["nbHashNeg"]=nbNegHash
    
    features["globalLexiconScore"]=globalLexiconScore
    features["nrcDecision"]=nrcDecision
    features["nrcEmoDecision"]=nrcEmoDecision
    features["nrcHashDecision"]=nrcHashDecision
    features["nrcScore"]=nrcScore #will keep
    features["nrcEmoScore"]=nrcEmoScore
    features["nrcHashScore"]=nrcHashScore
    features["globalLexiconDecision"]=globalLexiconDecision
    
    features["hasPosSmileys"]=hasPosSmileys
    features["hasNegSmileys"]=hasNegSmileys
    features["hasProfanity"]=hasProfanity
    features["nbOfPunctuation"]=nb_punct
    features["nbOfExclamation"]=nb_exclamation
    features['nbOfCapitalLetters']=len(upperCaseLetters)
    features['nbOfCapitalWords']=nbOfCapitalizedWords
    return features 

# <codecell>

#Initialize Set
trainSet=[]
trainSetX=[]
trainSetY=[]
trainSet2=[]
trainSet2X=[]
trainSet2Y=[]
testSet=[]
goldSetTA=[]
goldSetTAX=[]
goldSetTAY=[]
goldSetTBX=[]
goldSetTBY=[]
goldSetTB=[]
goldSetSA=[]
goldSetSAX=[]
goldSetSAY=[]
goldSetSB=[]
goldSetSBX=[]
goldSetSBY=[]

def makeSets():
    f = open('twitter_download-master/testCleansedAFinal2.tsv', 'rU')

    for line in f:
        formatedLine=line.split()
        lowerIndex=int(formatedLine[2])+5
        upperIndex=int(formatedLine[3])+5
        sentiment=formatedLine[4]
        tweet=' '.join(formatedLine[lowerIndex:upperIndex+1])
        trainSetX.append(tweet)
        trainSetY.append(sentiment)
        trainSet.append((twitter_features(tweet),sentiment))
    print "finished loading trainSetA"

    f = open('twitter_download-master/twitterTrainCleansedBFinal.tsv', 'rU')
    for line in f:
        formatedLine=line.split()
        sentiment=formatedLine[2]
        tweet=' '.join(formatedLine[3:])
        trainSet2X.append(tweet)
        trainSet2Y.append(sentiment)
        trainSet2.append((twitter_features(tweet),sentiment))
    print "finished loading trainSetB"

    f= open('twitter_download-master/twitterTestInputAFinal.tsv','rU')
    for line in f:
        formatedLine=line.split()
        lowerIndex=int(formatedLine[2])+5
        upperIndex=int(formatedLine[3])+5
        sentiment=formatedLine[4]
        tweet=' '.join(formatedLine[lowerIndex:upperIndex+1])
        testSet.append((twitter_features(tweet),sentiment))
    print "finished loading testSetA"

    f= open('twitter_download-master/twitterTestGoldAFinal.tsv','rU')
    for line in f:
        formatedLine=line.split()
        lowerIndex=int(formatedLine[2])+5
        upperIndex=int(formatedLine[3])+5
        sentiment=formatedLine[4]
        tweet=' '.join(formatedLine[lowerIndex:upperIndex+1])
        goldSetTAX.append(tweet)
        goldSetTAY.append(sentiment)
        goldSetTA.append((twitter_features(tweet),sentiment))
    print "finished loading goldTwitterSetA"

    f= open('twitter_download-master/twitterTestGoldBFinal.tsv','rU')
    for line in f:
        formatedLine=line.split()   
        sentiment=formatedLine[2]
        tweet=' '.join(formatedLine[3:])
        goldSetTBX.append(tweet)
        goldSetTBY.append(sentiment)
        goldSetTB.append((twitter_features(tweet),sentiment))
    print "finished loading goldTwitterSetB"

    f= open('twitter_download-master/sms-test-gold-A.tsv','rU')
    for line in f:
        formatedLine=line.split()
        lowerIndex=int(formatedLine[2])+5
        upperIndex=int(formatedLine[3])+5
        sentiment=formatedLine[4]
        tweet=' '.join(formatedLine[lowerIndex:upperIndex+1])
        goldSetSAX.append(tweet)
        goldSetSAY.append(sentiment)
        goldSetSA.append((twitter_features(tweet),sentiment))
    print "finished loading goldSmsSetA"

    f= open('twitter_download-master/sms-test-gold-B.tsv','rU')
    for line in f:
        formatedLine=line.split()    
        sentiment=formatedLine[2]
        tweet=' '.join(formatedLine[3:])
        goldSetSBX.append(tweet)
        goldSetSBY.append(sentiment)
        goldSetSB.append((twitter_features(tweet),sentiment))
    print "finished loading goldSmsSetB" 

# <codecell>


def trainLv1Classifiers():
    classifier = nltk.NaiveBayesClassifier.train(trainSet)
    classifier2 = nltk.NaiveBayesClassifier.train(trainSet2)
    return (classifier,classifier2)
def showLv1stats():
    print "Precision"
    print "Classifieur term"
    print "----------------"
    print "Twitter term" , nltk.classify.accuracy(classifier,goldSetTA)
    print "Twitter message" ,nltk.classify.accuracy(classifier,goldSetTB)
    print "SMS term" ,nltk.classify.accuracy(classifier,goldSetSA)
    print "SMS message" ,nltk.classify.accuracy(classifier,goldSetSB)
    print
    print "Classifieur message"
    print "----------------"
    print "Twitter term" , nltk.classify.accuracy(classifier2,goldSetTA)
    print "Twitter message" ,nltk.classify.accuracy(classifier2,goldSetTB)
    print "SMS term" ,nltk.classify.accuracy(classifier2,goldSetSA)
    print "SMS message" ,nltk.classify.accuracy(classifier2,goldSetSB)
    
    print "Features term classifier"
    print classifier.show_most_informative_features()
    print "Features message classifier"
    print classifier2.show_most_informative_features()
    
    sets=[goldSetTA,goldSetSA,goldSetTB,goldSetSB]
    for inputSet in sets:
        print "classifier term"
        scorer(inputSet,classifier)
        print "classifier message"
        scorer(inputSet,classifier2)
        print

# <codecell>

from sklearn import svm
def sumOpinionOfTweetInDict(tweet,dictionnary):
    tokens=nltk.word_tokenize(tweet)
    sumOfScores=0
    for token in tokens:
        if token in dictionnary:
            sumOfScores=sumOfScores+ dictionnary[token]
    return sumOfScores


def getOpinionVector(tweet,classifier):
    classifierOpinion= 1 if classifier.classify(twitter_features(tweet)) =='positive' else 0
    hashtagOpinion= sumOpinionOfTweetInDict(tweet,hashtagScoreDict)
    emoticonOpinion= sumOpinionOfTweetInDict(tweet,emoticonScoreDict)
    nrcOpinion=sumOpinionOfTweetInDict(tweet,nrcPosDict)-sumOpinionOfTweetInDict(tweet,nrcNegDict)
    return [classifierOpinion,hashtagOpinion,emoticonOpinion,nrcOpinion]

def showLv2Stats():

    scorerSVM(goldSetSAX,goldSetSAY,clf)
def trainClassifiers():
    classifier,classifier2=trainLv1Classifiers()
    f = open('twitter_download-master/testCleansedAFinal2.tsv', 'rU')
    devSetX=[]
    devSetY=[]
    for line in f:
        formatedLine=line.split()
        lowerIndex=int(formatedLine[2])+5
        upperIndex=int(formatedLine[3])+5
        sentiment=formatedLine[4]
        tweet=' '.join(formatedLine[lowerIndex:upperIndex+1])
        devSetX.append(getOpinionVector(tweet,classifier2))
        devSetY.append(sentiment)


    clf = svm.SVC()
    clf.fit(devSetX,devSetY)
    return(classifier,classifier2,clf)

# <codecell>

#Scorer
def scorer(inputSet ,classifier):
    totalOccurence={"positive":0 , "negative":0 , "neutral":0}
    foundOccurence={"positive":0 , "negative":0 , "neutral":0}
    classifyOccurence={"positive":0 , "negative":0 , "neutral":0}
    errorOccurence={"positive":0 , "negative":0 , "neutral":0}
    precision={"positive":0. , "negative":0. , "neutral":0.}
    recall={"positive":0. , "negative":0. , "neutral":0.}
    f={"positive":0. , "negative":0. , "neutral":0.}
    for i in xrange(len(inputSet)):
        totalOccurence[inputSet[i][1]]=totalOccurence[inputSet[i][1]]+1
        classifyOccurence[classifier.classify(inputSet[i][0])]=classifyOccurence[classifier.classify(inputSet[i][0])]+1

        if inputSet[i][1] != classifier.classify(inputSet[i][0]) :
            errorOccurence[inputSet[i][1]]=errorOccurence[inputSet[i][1]]+1
        else:
            foundOccurence[inputSet[i][1]]=foundOccurence[inputSet[i][1]]+1
    for key in foundOccurence.keys():
        precision[key]=float(foundOccurence[key])/float(classifyOccurence[key])
        recall[key]=float(foundOccurence[key])/float(totalOccurence[key])
        f[key]=(2*(precision[key]*recall[key]))/(precision[key]+recall[key])

#Scorer for the SVM prediction test        
def scorerSVM(setX,setY,predicter):   
    totalOccurence={"positive":0 , "negative":0 , "neutral":0}
    foundOccurence={"positive":0 , "negative":0 , "neutral":0}
    classifyOccurence={"positive":0 , "negative":0 , "neutral":0}
    errorOccurence={"positive":0 , "negative":0 , "neutral":0}
    precision={"positive":0. , "negative":0. , "neutral":0.}
    recall={"positive":0. , "negative":0. , "neutral":0.}
    f={"positive":0. , "negative":0. , "neutral":0.}
    for i in xrange(len(setY)):
        prediction=predicter.predict(getOpinionVector(setX[i]))[0]
        answer=setY[i]
        totalOccurence[answer]=totalOccurence[answer]+1
  
        classifyOccurence[prediction]=classifyOccurence[prediction]+1

        if answer != prediction :
            errorOccurence[answer]=errorOccurence[answer]+1
        else:
            foundOccurence[answer]=foundOccurence[answer]+1

    for key in foundOccurence.keys():
        precision[key]=float(foundOccurence[key]+1)/float(classifyOccurence[key]+1)
        recall[key]=float(foundOccurence[key]+1)/float(totalOccurence[key]+1)
        f[key]=(2*(precision[key]*recall[key])+1)/(precision[key]+recall[key]+1)

# <codecell>

#Testing against all provided goldSets
sets=[goldSetTA,goldSetSA,goldSetTB,goldSetSB]
for inputSet in sets:
    print "classifier term"
    scorer(inputSet,classifier)
    print "classifier message"
    scorer(inputSet,classifier2)
    print

# <codecell>

