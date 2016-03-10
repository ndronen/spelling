library(stringr)
library(ggplot2)
library(gtools)
library(Hmisc)
library(gridExtra)

#files <- dir("data", pattern="*-analysis.csv")
files <- dir(pattern="*-analysis.csv")

# Remove the Norvig corpus, which is a subset of some Mitton corpora, I believe.
#files <- paste0("data/", files)

# Load the first file.
file <- files[1]
df <- read.csv(file, sep='\t')

# Then the others.
for (f in files[-1]) {
  df <- rbind(df, read.csv(f, sep='\t'))
}

# Rename corpora and add length column.
df$corpus <- capitalize(df$corpus)
df$corpus <- str_replace(df$corpus, 'Holbrook-missp', 'Holbrook')
df$corpus <- factor(df$corpus,
  levels=c("Holbrook", "Wikipedia", "Aspell", "Birbeck"))
df$length <- str_length(df$non_word)

# Massage the column names for presentation.
colnames(df) <- str_replace_all(colnames(df), "_", " ")
colnames(df) <- capitalize(colnames(df))
print(names(df))

# The distributions of word lengths and of edit distances from the
# non-word to the true correction.
pdf("freq-dist-length-distance.pdf")
dist_length <- histogram(~Length | Corpus,
    data=df,
    xlab="Non-word length")
dist_levenshtein <- histogram(~`Levenshtein distance` | Corpus,
    data=df,
    xlab="Levenshtein distance to correction")
print(grid.arrange(dist_length, dist_levenshtein, nrow=2))
dev.off()

# The distributions of lengths of candidate lists -- need to do this
# for Aspell and Edit distance retrievers.
pdf("freq-dist-candidate-lists.pdf")
ed <- histogram(~`Edit distance n candidates` | Corpus,
          data=df,
          xlab="Near-miss retrieval")
aspell <- histogram(~`Aspell n candidates` | Corpus,
          data=df,
          xlab="Aspell retrieval",
          ylab=NULL)
print(grid.arrange(ed, aspell, ncol=2))
dev.off()

# For a given corpus and dictionary, what is the distribution of the
# ranks of true corrections in the candidate list for each error?

# What is that distribution conditioned on (1) the length of the non-word
# or (2) the edit distance from the non-word to its correction?
