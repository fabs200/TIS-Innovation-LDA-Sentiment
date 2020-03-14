#https://github.com/JBGruber/LexisNexisTools
# remotes::install_github("JBGruber/LexisNexisTools")

library("LexisNexisTools")
library("dplyr")
library("feather")
library("rlist")

rm(list = ls())

projectpath <- "/Users/Fabian/OneDrive/Projekt/Media Analysis/"


# define folder where the files are
filespath <- paste0(projectpath, "data/files")

articles <- list()

# Get all files in folder
filesfolder <- list.files(filespath, include.dirs = TRUE, full.names = TRUE)
nbrfolders = length(filesfolder)

# save all .docx to one list
for (i in 1:nbrfolders) {
  print(i)
  articles <- append(articles, list.files(filesfolder[i], include.dirs = TRUE, full.names = TRUE))
}

nbrarticles = length(articles)

#read in and transform docx articles
LNToutput <- lnt_read(x = articles[1])

# select only Publication-Type: Zeitung
LNToutput@meta$stats <- lnt_lookup(LNToutput, pattern = "Publication-Type: Zeitung")
#LNToutput <- LNToutput[!sapply(LNToutput@meta$stats, is.null), ]
LNToutput <- LNToutput[!sapply(LNToutput@meta$stats, is.null), ]

# convert LNT object to data.frame (Articles, Paragraphs)
df_articles <- lnt_convert(LNToutput, to = "data.frame", what = 'articles')
df_paragraphs <- lnt_convert(LNToutput, to = "data.frame", what = 'paragraphs')

for (i in 2:nbrarticles) {
  
  print(i)
  
  #read in and transform docx articles
  try(LNToutput <- lnt_read(x = articles[i]))
  
  # select only Publication-Type: Zeitung
  try(LNToutput@meta$stats <- lnt_lookup(LNToutput, pattern = "Publication-Type: Zeitung"))
  try(LNToutput <- LNToutput[!sapply(LNToutput@meta$stats, is.null), ])
  
  # convert LNT object to data.frame on articles level
  try(meta_article_temp <- lnt_convert(LNToutput, to = "data.frame", what = 'articles'))
  # replace ID by i
  try(meta_article_temp$ID <- i)
  
  # convert LNT object to data.frame on paragraph level
  try(meta_paragraph_temp <- lnt_convert(LNToutput, to = "data.frame", what = 'paragraphs'))
  # replace ID by i
  try(meta_paragraph_temp$Art_ID <- i)
  
  # add temporarily dfs to main dfs
  try(df_articles <- rbind(df_articles, meta_article_temp))
  try(df_paragraphs <- rbind(df_paragraphs, meta_paragraph_temp))

}

# Select variables
cols <- c("Source_File", "Newspaper", "Date", "Length", "Headline")
df_articles_export <- df_articles[ , c("ID", "Article", cols)]
df_paragraphs_export <- df_paragraphs[ , c("Art_ID", "Par_ID", "Paragraph", cols)]

# Export df_articles as feather/csv file
export <- "auto_articles_withbattery"
write_feather(df_articles_export, paste0(projectpath, "data/feather/", export, ".feather"))
write.csv(df_articles_export,paste0(projectpath, "data/csv/", export, ".csv"), row.names = FALSE)

# Export df_paragraphs as feather/csv file
export <- "auto_paragraphs_withbattery"
write_feather(df_paragraphs_export, paste0(projectpath, "data/feather/", export, ".feather"))
write.csv(df_paragraphs_export,paste0(projectpath, "data/csv/", export, ".csv"), row.names = FALSE)

