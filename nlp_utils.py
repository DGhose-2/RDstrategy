import pandas as pd
import numpy as np

def presplit_preprocess(df):
    # remove companies with less than 4 projects
    df_red = df[df['count'] >= 4]

    # remove unnecessary columns
    # at this point we only need orguuid (unique identifier of org), (project) title, (organization) description, CrunchBase Rank. (Later, cost and cost-offer: this needs to be imputed for negative examples)
    df_red = df_red[['orguuid', 'description', 'rank_x', 'title', 'count', 'startdate', 'enddate']].rename(columns={'rank_x': 'CB_rank', 'count': 'projects_count'})

    # datetime formats for project-dates and engineering project length
    df_red[['startdate','enddate']] = df_red[['startdate','enddate']].apply(pd.to_datetime)
    df_red['project_length'] = df_red.apply(lambda row: row['enddate'] - row['startdate'], axis=1)
    df_red = df_red.drop('enddate', axis=1)
    
    df_red = df_red[~df_red.duplicated(subset=['orguuid', 'title'])]
    
    return df_red

def trainval_test_split(df, split_frac=0.75):
    # Splitting into training-validation and test sets
    trainval_frac = split_frac
    
    # arrange to get only the last count-CEIL(trainval_frac*count) projects in test set
    df_trainval = df.groupby(['orguuid','projects_count']).head(np.ceil(df['projects_count']*trainval_frac).round().astype(int))
    df_test = df[~(df.index.isin(df_trainval.index))]
    
    # reset indexes
    df_trainval = df_trainval.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # attach labels to trainval (for negative example generation)
    df_trainval['Occurred'] = 1
    
    # for identifying training-val apart from test
    df_trainval['typ'] = 'tv'
    return df_trainval, df_test

def trainval_negs(df, full_data):
    # Generating negative examples for training-validation set

    pairs = list(full_data[['orguuid', 'title']].itertuples(index=False, name=None))

    orgs = full_data.orguuid.unique()
    projs = full_data.title.unique()

    import random

    neg_size = len(df)
    neg_batch = []
    neg_id=0
    neg_label=0

    while neg_id < neg_size:
        random_org = orgs[random.randrange(len(orgs))]
        random_proj = projs[random.randrange(len(projs))]

        if (random_org, random_proj) not in pairs and (random_org, random_proj, neg_label) not in neg_batch:
            neg_batch.append((random_org, random_proj, neg_label))
            neg_id += 1

    # Now add this list to the trainval dataframe
    df_neg = pd.DataFrame(neg_batch, columns=['orguuid', 'title', 'Occurred'])
    
    # left-join with information on companies from the full dataset
    df_neg_filled = df_neg.reset_index().merge(full_data, left_on=['orguuid'], right_on=['orguuid'], how="left").set_index("index").drop_duplicates(subset=["orguuid", "title_x"])
    df_neg_filled = df_neg_filled[['orguuid', 'description', 'CB_rank', 'title_x', 'projects_count', 'Occurred']]
    
    # left-join with information on projects from the full dataset
    df_neg_filled = df_neg_filled.reset_index().merge(full_data, left_on=['title_x'], right_on=['title'], how="left").set_index("index").drop_duplicates(subset=["orguuid_x", "title_x"])
    df_neg_filled = df_neg_filled[['orguuid_x', 'description_x', 'CB_rank_x', 'title_x', 'projects_count_x', 'startdate', 'project_length', 'Occurred']]
    
    # add label to match trainval label
    df_neg_filled['typ'] = 'tv'
    
    # rename columns to match those of trainval
    df_neg_filled.columns = df.columns

    # add the negative examples to the positive ones
    df = pd.concat([df, df_neg_filled], axis=0).reset_index(drop=True)
    
    return df

def test_combs(df, col_order):
    # Generating all combinations for the test data

    org_cols = ['description', 'CB_rank', 'projects_count']
    proj_cols = ['startdate', 'project_length']

    mux = pd.MultiIndex.from_product([df.orguuid.unique(), df.title.unique()], names=('orguuid','title'))
    df = df.set_index(['orguuid','title']).reindex(mux).reset_index()
    df['Occurred'] = df[org_cols[0]].notnull().astype(int)
    df[org_cols] = df.groupby('orguuid')[org_cols].transform('first')  # get the organization info back
    df[proj_cols] = df.groupby('title')[proj_cols].transform('first') # get the project info back

    #label for identifying test
    df['typ'] = 'tst' 
    df = df[col_order].reset_index(drop=True) #put columns in the same order as trainval, for easy concatenation later
    
    return df

# similarity scores, e.g. with doc2Vec

def similarity_scores(df, meth):
    from collections import defaultdict
    from gensim import corpora

    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    textsList = df[['description', 'title']].values.T.tolist()
    textsList_flat = [item for sublist in textsList for item in sublist]

    documents = textsList_flat
    
    if meth=='d2v':
        # Doc2Vec preprocessing
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(documents)] #this is sufficient for a word-order--conserving model where we will retain punctuation
        
        #Doc2Vec

        #run the model
        max_epochs = 10
        vec_size = 25
        alpha = 0.03

        model = Doc2Vec(size=vec_size,            
                        alpha=alpha,              
                        min_alpha=0.00025,        
                        min_count = 2,            
                        dm = 1
        )

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.00025
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        model.save("d2v_mixed7.model")
        print("Model Saved")
        
        # compute cosine similarity
        cossims = []
        for i in range(len(df)):
            cossimil = model.docvecs.similarity(i, (len(df)+i))
            cossims.append(cossimil)
    
    elif meth=='lda' or meth=='lsi':
        # BOW model preprocessing
        
        # Split the document into tokens
        from gensim.utils import simple_preprocess
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(simple_preprocess(str(sentence), deacc=True))
        texts = list(sent_to_words(documents))

        # Remove common words and words that are only one character.
        stoplist = set('for a of the and to in'.split())
        #stoplist = set(stopwords.words('english'))
        texts = [[token for token in doc if (len(token)>1) and (token not in stoplist)] for doc in texts]

        # Lemmatize the documents.
        from nltk.stem.wordnet import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        texts = [[lemmatizer.lemmatize(token) for token in doc] for doc in texts]

        # Lemmatized reduction with spaCy package, to keep only certain word-classes (noun, adjective, verb, adverb) i.e. remove prepositions etc
        # function from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            texts_red = []
            for sentence in texts:
                doc = nlp(" ".join(sentence)) 
                texts_red.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_red

        import spacy
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        texts = lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        # Filter carefully to remove rarest words (occurring in less than 15 documents), or common lemmas (more than 60% of the documents)
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below = 15, no_above = 0.6)

        # Construct the final corpus, bag-of-words representation of documents
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        #Run the model
        if meth=='lda':

            #LDA
            
            from gensim import models

            lda_model = models.LdaModel(corpus=corpus,
                                        id2word=dictionary,
                                        alpha='auto',
                                        eta='auto',
                                        iterations=10,
                                        passes=2,
                                        num_topics=100,
                                        eval_every=None,
                                        decay = 0.8,
                                        offset = 1
                                       )
            corpus_lda = lda_model[corpus]
            
            # compute cosine similarity
            from gensim.matutils import cossim 
            cossims = []

            for i in range(len(df)):
                cossimil = cossim(corpus_lda[i], corpus_lda[len(df)+i])
                cossims.append(cossimil)
        
        else:
            
            #LSI (with TFIDF)
            
            from gensim import models

            tfidf = models.TfidfModel(corpus,
                                      smartirs='npc', #probabilistic idf
                                      slope=0.2)      #lower slope means longer documents are favoured more (usually an effective choice for TFIDF)
            corpus_tfidf = tfidf[corpus]
            lsi_model = models.LsiModel(corpus_tfidf,
                                        id2word=dictionary,
                                        num_topics=300,
                                        power_iters=2
                                        )
            corpus_lsi = lsi_model[corpus_tfidf]
            
            # compute cosine similarity
            from gensim.matutils import cossim 
            cossims = []
            
            for i in range(len(df)):
                cossimil = cossim(corpus_lsi[i], corpus_lsi[len(df)+i])
                cossims.append(cossimil)

    else:
        print("Please provide a valid method ('lda', 'lsi', 'd2v')")
        
    df_sims = df.assign(sim = cossims)
    return df_sims