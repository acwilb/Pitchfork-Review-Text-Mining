library(dplyr) # data reading + cleaning
library(tm)
library(tidyr)
library(ggplot2) # visualizations
library(RColorBrewer)
library(Rgraphviz)
library(wordcloud2)
library(igraph)
library(ggraph)
library(gridExtra)
library(syuzhet) # sentiments
library(tidytext)
library(cluster) # cluster analysis
library(factoextra)
library(topicmodels) # lda
library(caTools) # predicitve modeling
library(randomForest)

setwd("~/STATS 133/Project")

# Reading in Data ---------------------------------------------------------

pitchfork <- read.csv("pitchfork_reviews.csv")
pitchfork <- pitchfork %>% filter(score != "Not Available") %>% filter(genre != "Not Available")

# EDA ---------------------------------------------------------------------

pitchfork$score <- as.numeric(as.vector(pitchfork$score))
summary(pitchfork$score)

# Barplot - Distribution of Score
as.data.frame(pitchfork) %>% mutate(med = median(score)) %>% 
  ggplot(aes(x = score, y = med)) + 
  geom_col(fill = "thistle4") + 
  labs(x = "Album Score", y = "Frequency")

# Categorize multiple-genre albums into the one genre (e.g. Folk/Country becomes Folk)
pitchfork$genre.single <- as.character(pitchfork$genre)
for (i in 1:nrow(pitchfork)){
  if (grepl("/", pitchfork$genre.single[i]) == TRUE){
    pitchfork$genre.single[i] <- gsub("/.*$", "", pitchfork$genre.single[i])
  }
}

# Barplot - Distribution of Median Score per Genre
pitchfork$genre.single <- factor(pitchfork$genre.single, levels = c("Electronic", "Experimental", "Folk", "Global", "Jazz", "Metal", "Pop", "Rap", "Rock"))
na.omit(pitchfork) %>% group_by(genre.single) %>% summarise(med = median(score))
as.data.frame(na.omit(pitchfork)) %>% group_by(genre.single) %>% summarise(med = median(score)) %>% 
  ggplot(aes(x = genre.single, y = med, fill = genre.single)) + 
  geom_bar(stat="identity") +
  scale_fill_manual(values = c("sienna", "seagreen", "lightpink4", "lightslategrey", "khaki4", "tomato4", "pink4", "thistle4", "indianred")) +
  labs(x = "Genre", y = "Median Score") + theme(legend.position = "none") + coord_cartesian(ylim = c(6, 8))

# Preprocessing Text Data -------------------------------------------------------------

# Create text corpus
pitchfork.corp <- VCorpus(VectorSource(pitchfork$review))
pitchfork.corp <- tm_map(pitchfork.corp, PlainTextDocument)

# Cleaning corpus
pitchfork.corp <- tm_map(pitchfork.corp, content_transformer(tolower)) # convert to lowercase
pitchfork.corp <- tm_map(pitchfork.corp, removePunctuation) # remove punctuation
pitchfork.corp <- tm_map(pitchfork.corp, removeWords, stopwords("english")) # remove stopwords
pitchfork.corp <- tm_map(pitchfork.corp, removeWords, c("album", "music", "songs", "song")) # remove own stopwords
pitchfork.corp <- tm_map(pitchfork.corp, stripWhitespace) # strip whitespace
pitchfork.corp <- tm_map(pitchfork.corp, stemDocument) # stemming


# Word Associations -------------------------------------------------------

pitchfork.dtm <- DocumentTermMatrix(pitchfork.corp)
pitchfork.dtm.s <- removeSparseTerms(pitchfork.dtm, 0.90)

# Correlation Plot
plot(pitchfork.dtm, terms = findFreqTerms(pitchfork.dtm, lowfreq = 500), corThreshold = 0.80)

# Word Frequency ----------------------------------------------------------

# df with text
pitchfork.df1 <- tibble(line = 1:nrow(pitchfork), text = pitchfork$review)
own_stop_words <- tibble(word = c("album", "music", "songs", "song", "it's"))
tidy.pitchfork <- pitchfork.df1 %>% unnest_tokens(word, text) %>% anti_join(stop_words) %>% anti_join(own_stop_words)

# plot most frequent words
tidy.pitchfork %>% count(word, sort = TRUE) %>% filter(n > 10000) %>% mutate(word = reorder(word, n)) %>% 
  ggplot(aes(word, n)) + geom_col(fill = "thistle4") + labs(x = NULL, y = "Frequency") + coord_flip()

# df with genre
pitchfork.df <- tibble(line = 1:nrow(pitchfork), genre = pitchfork$genre.single, text = pitchfork$review)

# most frequent words by genre
pitchfork.words <- pitchfork.df %>% unnest_tokens(word, text) %>% anti_join(stop_words) %>% count(genre, word, sort = TRUE)

# term frequency by genre
total.words <- pitchfork.words %>% group_by(genre) %>% summarize(total = sum(n))
genre.words <- left_join(pitchfork.words, total.words)
genre.word.ratios <- genre.words$n/genre.words$total
# plotting tf by genre
genre.words %>% ggplot(aes(genre.word.ratios, fill = genre)) + 
  geom_histogram(show.legend=FALSE) + 
  scale_fill_manual(values = c("sienna", "seagreen", "lightpink4", "lightslategrey", "khaki4", "tomato4", "pink4", "thistle4", "indianred")) + 
  xlim(NA, 0.0009) + 
  xlab("Genre TF") + facet_wrap(~genre, ncol = 3, scales = "free_y")

# tf-idf by genre
genre.words.tfidf <- genre.words %>% bind_tf_idf(word, genre, n)
# plotting top 10 tf-idf by genre
genre.words.tfidf %>% select(-total) %>% arrange(desc(tf_idf)) %>% 
  mutate(word = factor(word, levels = rev(unique(word)))) %>% 
  group_by(genre) %>% top_n(10) %>% 
  ungroup() %>% 
  ggplot(aes(word, tf_idf, fill = genre)) + geom_col(show.legend = FALSE) +
  scale_fill_manual(values = c("sienna", "seagreen", "lightpink4", "lightslategrey", "khaki4", "tomato4", "pink4", "thistle4", "indianred")) + 
  labs(y = "tf-idf") + facet_wrap(~genre, ncol=3, scales = "free") + coord_flip()

# word cloud
set.seed(12345)
word.df <- data.frame(tidy.pitchfork %>% count(word))
wordcloud2(word.df, minSize = 20)

# Sentiment Analysis ------------------------------------------------------

pitchfork.df.sent <- tibble(line = 1:nrow(pitchfork), text = pitchfork$review)
bing <- get_sentiments("bing")
pitchfork.sentiments <- pitchfork.df.sent %>% unnest_tokens(word, text) %>% inner_join(bing)

# Bing positive & negative
sent.positive <- bing %>% filter(sentiment == "positive")
sent.negative <- bing %>% filter(sentiment == "negative")

# Plotting distribution of positive & negative reviews
pos.neg <- rbind(pitchfork.sentiments %>% inner_join(sent.negative), pitchfork.sentiments %>% inner_join(sent.positive))
pos.neg %>% ggplot(aes(x = sentiment, y = "", fill = sentiment)) + 
  geom_col(show.legend = FALSE) + scale_fill_manual(values = c("tomato4", "seagreen")) + labs(x = "Sentiment", y = "Proportion")

# top 10 most frequent negative terms
neg.freq <- pitchfork.sentiments %>% inner_join(sent.negative) %>% count(word, sort = TRUE) %>% top_n(10)
neg.freq %>% ggplot(aes(x = word, y = n)) + geom_col(fill = "tomato4") + xlab(NULL) +ylab("Frequency") + coord_flip()

# top 10 most frequent positive terms
pos.freq <- pitchfork.sentiments %>% inner_join(sent.positive) %>% count(word, sort = TRUE) %>% top_n(10)
pos.freq %>% ggplot(aes(x = word, y = n)) + geom_col(fill = "seagreen4") + xlab(NULL) +ylab("Frequency") + coord_flip()

# N-Grams -----------------------------------------------------------------

# Bigrams
pitchfork.bigrams <- pitchfork.df1 %>% unnest_tokens(bigram, text, token = "ngrams", n = 2)
pitchfork.bigrams

# Counting and Filtering
pitchfork.bigrams %>% count(bigram, sort = TRUE) %>% filter(n > 10000) %>% ggplot(aes(bigram, n)) + geom_col() + xlab(NULL) + ylab("Count") + coord_flip()

# Avoiding stop words
bigrams.separated <- pitchfork.bigrams %>% separate(bigram, c("word1", "word2"), sep = " ")
bigrams.filtered <- bigrams.separated %>% filter(!word1 %in% stop_words$word) %>% filter(!word2 %in% stop_words$word)

# New bigram counts:
bigram.counts <- bigrams.filtered %>% count(word1, word2, sort = TRUE) 
bigrams.united <- bigrams.filtered %>% unite(bigram, word1, word2, sep = " ") 
bigrams.u.count <- bigrams.united %>% count(bigram, sort = TRUE)
bigrams.u.count %>% filter(n > 1000) %>% ggplot(aes(bigram, n)) + geom_col(fill = "thistle4") + labs(x = NULL, y = "Frequency") + coord_flip()

# tf-idf
bigram.tf_idf <- bigrams.united %>% count(line, bigram) %>% bind_tf_idf(bigram, line, n) %>% arrange(desc(tf_idf))
bigram.tf_idf

# Negation bigrams
AFINN <- get_sentiments("afinn")
not.words <- bigrams.separated %>% filter(word1 == "not") %>% 
  inner_join(AFINN, by = c(word2 = "word")) %>% count(word2, value, sort = TRUE)
not.words %>% mutate(contribution = n * value) %>% 
  arrange(desc(abs(contribution))) %>% head(20) %>% 
  mutate(word2 = reorder(word2, contribution)) %>% 
  ggplot(aes(word2, n * value, fill = n * value > 0)) + 
  geom_col(show.legend = FALSE) + xlab("Words preceded by \"not\"") + 
  scale_fill_manual(values = c("tomato4", "seagreen")) + 
  ylab("Sentiment value * number of occurrences") + coord_flip()

# visualizing bigrams network
set.seed(12345)
bigram.graph <- bigram.counts %>% filter(n > 300) %>% graph_from_data_frame() 
ggraph(bigram.graph, layout = "fr") + geom_edge_link() +geom_node_point() +geom_node_text(aes(label = name), vjust = 1, hjust = 1)


# LDA ---------------------------------------------------------------------

zero.entries <- which(apply(pitchfork.dtm[, 1], 1, sum) == 0)
pitchfork.lda <- LDA(pitchfork.dtm[-zero.entries, ], k = 5, control = list(seed = 12345))

# Isolate topics
pitchfork.topics <- tidy(pitchfork.lda, matrix = "beta")

# Finding the Top Ten Terms
pitchfork.top.terms <- pitchfork.topics %>% group_by(topic) %>% top_n(10, beta) %>% ungroup() %>% arrange(topic, -beta)

# Plot the Top Terms for each Topic
pitchfork.top.terms %>% mutate(term = reorder_within(term, beta, topic)) %>% 
  ggplot(aes(term, beta, fill = factor(topic))) + geom_col(show.legend = FALSE) + 
  scale_fill_manual(values = c("sienna", "seagreen", "lightpink4", "lightslategrey", "khaki4")) +
  facet_wrap(~topic, scales = "free") + coord_flip() + scale_x_reordered()


# Cluster Analysis --------------------------------------------------------

nrc <- get_nrc_sentiment(as.character(pitchfork$review))
pitchfork$nrc_negative <- nrc$negative
pitchfork$nrc_positive <- nrc$positive

vizdata <- as.data.frame(as.vector(pitchfork))
vizdata <- vizdata[, c(3, 13, 14)]
vizdata.scaled <- scale(vizdata)

# Use gap statistic method for optimal number of clusters
gapstat <- clusGap(vizdata.scaled, FUN = kmeans, nstart = 25, K.max = 9, B = 50)
fviz_gap_stat(gapstat)

pitchfork.cluster <- kmeans(vizdata.scaled, 2, nstart=25)
pitchfork.cluster
# visualizing the cluster
fviz_cluster(pitchfork.cluster, data = vizdata) + 
  scale_colour_manual(values = c("seagreen", "tomato4")) + 
  scale_fill_manual(values = c("seagreen", "tomato4"))

vizdata %>% mutate(Cluster = pitchfork.cluster$cluster) %>% group_by(Cluster) %>% summarise_all("median")


# Predicitve Modeling -----------------------------------------------------

# categorizing response variable into binary options
pitchfork$highscore <- rep(NA, nrow(pitchfork))
for (i in 1:nrow(pitchfork)){
  if (pitchfork$score[i] > 7.3){
    pitchfork$highscore[i] <- 1
  } else {
    pitchfork$highscore[i] <- 0
  }
}

# Word Frequencies
pitchfork.sparse <- as.data.frame(as.matrix(pitchfork.dtm.s)) # convert format of sparse dtm
colnames(pitchfork.sparse) <- make.names(colnames(pitchfork.sparse)) # change variable names
pitchfork.sparse$highscore <- pitchfork$highscore

# Baseline accuracy
prop.table(table(pitchfork$highscore))
# 47.8% accuracy

# Split data into Training and Testing 
set.seed(12345)
split <- sample.split(pitchfork.sparse$highscore, SplitRatio = 0.7)
trainSparse <- subset(pitchfork.sparse, split == TRUE)
testSparse <- subset(pitchfork.sparse, split == FALSE)
dim(trainSparse)
dim(testSparse)
prop.table(table(trainSparse$highscore))
prop.table(table(testSparse$highscore))

# Random Forest
set.seed(12345)
trainSparse$highscore <- as.factor(trainSparse$highscore)
testSparse$highscore <- as.factor(testSparse$highscore)
pitchforkRF <- randomForest(highscore ~., data = trainSparse, ntree = 50)

predictRF <- predict(pitchforkRF, newdata = testSparse)
table(testSparse$highscore, predictRF)
#4355/6829=0.63772148
# RF model accuracy: 0.63772148, 63.8%

