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
LNToutput <- LNToutput[!sapply(LNToutput@meta$stats, is.null), ]

# convert LNT object to data.frame
main_df <- lnt_convert(LNToutput, to = "data.frame")


for (i in 2:nbrarticles) {
  
  print(i)
  
  #read in and transform docx articles
  try(LNToutput <- lnt_read(x = articles[i]))
  
    # select only Publication-Type: Zeitung
    try(LNToutput@meta$stats <- lnt_lookup(LNToutput, pattern = "Publication-Type: Zeitung"))
  try(LNToutput <- LNToutput[!sapply(LNToutput@meta$stats, is.null), ])
    
    # convert LNT object to data.frame
  try(meta_article_temp <- lnt_convert(LNToutput, to = "data.frame"))
  # replace ID by i
  try(meta_article_temp$ID <- i)
  
    # add to main_df
  try(main_df <- rbind(main_df, meta_article_temp))

}

df <- cbind(main_df[, 1:10], main_df[, 12])
df_copy <- df

#names(df)
#df = df[order(df[,'Date'],-df[,'Depth']),]
#df = df[!duplicated(df$Date),]

# Export Test file
export <- "autofiles_withbattery"
write_feather(df, paste0(projectpath, "data/feather/", export, ".feather"))
write.csv(df,paste0(projectpath, "data/csv/", export, ".csv"), row.names = FALSE)

