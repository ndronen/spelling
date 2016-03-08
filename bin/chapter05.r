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

# The distributions of word lengths.
#pdf("freq-dist-non-word-length.pdf")
#print(histogram(~Length | Corpus, data=df, xlab="Non-word length"))
#dev.off()

# The distributions of edit distances from the non-word to the true correction.
#pdf("freq-dist-levenshtein-distance.pdf")
#print(histogram(~`Levenshtein distance` | Corpus, data=df))
#dev.off()

# The distributions of lengths of candidate lists -- need to do this
# for Aspell and Edit distance retrievers.
graphics.off()
pdf("freq-dist-candidate-lists.pdf")
ed <- histogram(~`Edit distance n candidates` | Corpus,
          data=df,
          xlab="Length of candidate list (Edit distance)")
aspell <- histogram(~`Aspell n candidates` | Corpus,
          data=df,
          xlab="Length of candidate list (Aspell)",
          ylab=NULL)
print(grid.arrange(ed, aspell, ncol=2))
dev.off()
