import libspacy
import libngram

def identify_features(positive, negative, max_features=100):
    print ("Entered identify_features")
    word_counts={}
    num_positive = len(positive)
    num_negative = len(negative)
    num_positive = 1 
    num_negative = 1
    for sentence in positive:
        sentence=sentence.lower()
        words = sentence.split(' ')
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word,0) + num_negative
    #Do it for bigrams
    for sentence in positive:
        sentence=sentence.lower()
        words = libngram.ngrams_text(sentence, 2)
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word,0) + num_negative
    #Do it for trigrams
    for sentence in positive:
        sentence=sentence.lower()
        words = libngram.ngrams_text(sentence, 3)
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word,0) + num_negative

    #Subtract for negative class
    for sentence in negative:
        sentence=sentence.lower()
        words = sentence.split(' ')
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word,0) - num_positive
    #Do it for bigrams
    for sentence in positive:
        sentence=sentence.lower()
        words = libngram.ngrams_text(sentence, 2)
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word,0) - num_positive
    #Do it for trigrams
    for sentence in positive:
        sentence=sentence.lower()
        words = libngram.ngrams_text(sentence, 3)
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word,0) - num_positive


    #Update those words whose counts are negative to positive
    for (word, count) in word_counts.items():
        if count < 0:
           word_counts[word] = -count


    #Now sort the word counts by descending order of counts

    sorted_words = sorted(word_counts.items(), key = lambda x: x[1], reverse=True)
    #print sorted_words[:100]
    print ("Exiting identify_features")
    return [ i for i in sorted_words[:max_features]]

def generate_features(instance, features):
    X=[]
    for feature in features:
        #print feature
        if feature in instance:
            X.append(1)
        else:
            X.append(0)

    #Append the word vector
    #vector = libspacy.get_vector(instance)
    #X.extend(vector)
    return X
def parse_features(features):
    return [ libspacy.nlp(feature.decode('utf-8')) for feature in features]

def generate_features2(instance, features):
    X=[]
    sentence = libspacy.nlp(instance.decode('utf-8'))
    max_sim = 0.0
    for feature in features:
        for word in sentence:
            sim = word.similarity(feature)
            print ("Similarity of %s and %s is %f" % (word.text, feature.text, sim))
            if sim >=max_sim:
                max_sim = sim
                max_feat = word.text
        print ("Max similarity of %s is with %s" % (feature.text, word.text))
        X.append(max_sim)
    return X


