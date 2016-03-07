library(stringr)
library(ggplot2)
library(gtools)
library(Hmisc)

files <- dir("data", pattern="*-analysis.csv")

# Remove the Norvig corpus, which is a subset of some Mitton corpora, I believe.
files <- files[-which(grepl("norvig", files))]
files <- paste0("data/", files)
print(files)

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
df$length <- str_length(df$non_word)

# Massage the column names for presentation.
colnames(df) <- str_replace(colnames(df), "_", " ")
colnames(df) <- capitalize(colnames(df))

pdf("freq-dist-non-word-length.pdf")
histogram(~Length | Corpus, data=df, xlab="Non-word length")
dev.off()

pdf("freq-dist-levenshtein-distance.pdf")
histogram(~`Levenshtein distance` | Corpus, data=df)
dev.off()
