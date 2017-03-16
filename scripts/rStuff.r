
setwd("/home/jeffrey/Dropbox/Grad School/research/ranking/data/MQ2007/Fold2")
letor=read.table("letor.data",sep=",")

library("randomForest")

rf.model <- randomForest(
        letor,
        keep.forest = TRUE,
        ntree       = 100,
        nodesize    = 42,
        proximity   = FALSE)
