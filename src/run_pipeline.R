library(imager)
library(class)
library(glmnet)

cat("Project started...\n")

image_dir <- "data"

files <- list.files(image_dir, full.names = TRUE)

cat("Number of images:", length(files), "\n")