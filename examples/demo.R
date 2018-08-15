library(data.table)

SIZE <- 10000000

start.time <- Sys.time()
df = data.table(ints=1:SIZE, floats=1:SIZE + 0.5,bools=1:SIZE %% 2 == 0, paste("number ", 1:SIZE))
elapsed <- Sys.time() - start.time
print(paste("Created frame in ", as.numeric(elapsed)))

start.time <- Sys.time()
summaries = summary(df)
elapsed <- Sys.time() - start.time
print(paste("Summarized in ", as.numeric(elapsed)))
