import logging
from gensim.models import TfidfModel, LsiModel, LdaModel
from gensim import corpora

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# 读取字典和词袋
dictionary_en = corpora.Dictionary.load(r"Data\Intermediate\dict_en.dict")
corpus_en = corpora.BleiCorpus(r"Data\Intermediate\corpus_en.blei")

# tfidf模型
tfidf = TfidfModel(corpus_en)
tfidf.save(r"Data\Output\tfidf_model_en")

# 用tfidf模型计算tfidf的语料
tfidf_corpus = tfidf[corpus_en]
corpora.BleiCorpus.serialize(r"Data\Intermediate\tfidf_corpus_en.blei", tfidf_corpus)

# lsi
lsi = LsiModel(corpus=tfidf_corpus, id2word=dictionary_en, num_topics=20)
lsi_corpus = lsi[tfidf_corpus]
lsi.save(r"Data\Output\lsi_model_en")
corpora.BleiCorpus.serialize(r"Data\Intermediate\lsi_corpus_en.blei", lsi_corpus)
lsi.print_topics()

# lda
lda = LdaModel(corpus=tfidf_corpus, id2word=dictionary_en, num_topics=20)
lda_corpus = lda[tfidf_corpus]
lda.save(r"Data\Output\lda_model_en")
corpora.BleiCorpus.serialize(r"Data\Intermediate\lda_corpus_en.blei", lda_corpus)
lda.print_topics()
