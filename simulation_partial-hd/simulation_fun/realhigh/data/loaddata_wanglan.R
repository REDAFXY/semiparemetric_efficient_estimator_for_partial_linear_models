
#library("lumi")
#??lumi



# Unable to generate script analyzing differential expression.
#      Invalid input: at least two groups of samples should be selected.

################################################################
#   Boxplot for selected GEO samples
setwd("F:/MY FILE/simulation/partial linear - final/1022/realdata-high")
library(Biobase)
library(GEOquery)

# load series and platform data from GEO

gset <- getGEO("GSE27272", GSEMatrix =TRUE, getGPL=FALSE)
if (length(gset) > 1) idx <- grep("GPL6883", attr(gset, "names")) else idx <- 1
gset <- gset[[idx]]

# set parameters and draw the plot

dev.new(width=4+dim(gset)[[2]]/5, height=6)
par(mar=c(2+round(max(nchar(sampleNames(gset)))/2),4,2,1))
title <- paste ("GSE27272", '/', annotation(gset), " selected samples", sep ='')
boxplot(exprs(gset), boxwex=0.7, notch=T, main=title, outline=FALSE, las=2)


#summary:jianshu.com/p/0a24e8317484

#alldata
gset:
  #gen data
  exprs(gset)
# mei hang xingxi
pdata<-pData(gset)
head(pdata)
# zhushi
fdata<-fData(gset)
head(fdata)


#write data:   https://blog.csdn.net/Magic_Ninja/article/details/80846996

listnum <- list(gset$`age (years):ch1`,gset$`apgar score (5s):ch1`,gset$`cord blood cotinine (ng/ml):ch1`,gset$`gestational age (weeks):ch1`,gset$`individual:ch1`,
                gset$`maternal blood cotinine (ng/ml):ch1`,gset$`maternal bmi:ch1`,gset$`mode of delivery:ch1`,gset$`newborn weight (g):ch1`,gset$`placental weight (g):ch1`
                ,gset$`smoking status:ch1`,gset$`tissue:ch1`,gset$`parity:ch1`)
names(listnum) <- c('age (years)','apgar score (5s)','cord blood cotinine (ng/ml)','gestational age (weeks)','individual',
                    'maternal blood cotinine (ng/ml)','maternal bmi','mode of delivery','newborn weight (g)','placental weight (g)','smoking status','tissue','parity')

listall = list(gset$characteristics_ch1.1,gset$characteristics_ch1.2,gset$characteristics_ch1.3,gset$characteristics_ch1.4,gset$characteristics_ch1.5,gset$characteristics_ch1.6,gset$characteristics_ch1.7,
               gset$characteristics_ch1.8,gset$characteristics_ch1.9,gset$characteristics_ch1.10,gset$characteristics_ch1.11,gset$characteristics_ch1.12)
listchange <- listall
listchange[[2]] <- as.numeric(substr(listall[[2]], 14,15))
listchange[[3]] <- as.numeric(substr(listall[[3]], 15,20))
listchange[[5]] <- as.numeric(substr(listall[[5]], 26,27))
listchange[[7]] <- as.numeric(substr(listall[[7]], 23,25))
listchange[[8]] <- as.numeric(substr(listall[[8]], 21,23))
listchange[[10]] <- as.numeric(substr(listall[[10]], 34,40))
listchange[[11]] <- as.numeric(substr(listall[[11]], 30,36))
#age 2/ bmi 3/ preg age 5 / placental weight 7/ newborn weight 8 /maternal blood cotinine 10/ cord blood cotinine 11
write.csv(exprs(gset), file = "gendata.csv")
write.csv(listall, file = "chdata.csv")
write.csv(listchange, file = "change-chdata.csv")
write.csv(listnum, file = "allnumdata.csv")
