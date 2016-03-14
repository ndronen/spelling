library(stringr)
library(ggplot2)
library(gtools)
library(Hmisc)
library(gridExtra)

plot_analysis <- function() {
  dir.create("figures/chapter05", recursive=TRUE, showWarnings=FALSE)

  files <- dir("data", pattern="*-analysis.csv")
  files <- paste0("data/", files)
  
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

  pdf("figures/chapter05/freq-dist-length-distance.pdf")
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
  pdf("figures/chapter05/freq-dist-candidate-lists.pdf")
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
}

plot_density_of_logprob_by_length <- function() {
  library(ggplot2)
  library(stringr)

  dir.create("figures/chapter06/", recursive=TRUE, showWarnings=FALSE)
  
  df <- read.csv('data/aspell-dict.csv.gz', sep='\t')
  df$Length <- str_length(df$word)
  lengths <- c(2,3,4,5,7,9,11,13,15,17,19)
  df <- subset(df, Length %in% lengths)
  df$Length <- as.factor(df$Length)
  df$logprob <- log(df$google_ngram_prob)
  m <- ggplot(df, aes(x=logprob, color=Length))
  m <- m + geom_density()
  m <- m + xlim(-60, 0)
  m <- m + labs(x="Log probability", y="Density")
  ggsave("figures/chapter06/logprob-density-by-length.pdf", m)
}

plot_density_of_logprob_by_length()
plot_analysis()
