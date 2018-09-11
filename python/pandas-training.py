import pandas as pd
import time
import numpy as np

#------------------------------

begin = time.time()
df = pd.read_csv("train.csv")
#df = pd.read_csv("prediction.csv")
print("dataset loaded in ",time.time()-begin," seconds")

print("top 10 rows of dataset: ")
print(df.head(10))

print("dataset has following columns: ",df.columns)

print("Campaign_ID column has ",df['Campaign_ID'].dtypes," datatype. It can be following variables:")
print(df['Campaign_ID'].unique())


for index, row in df.iterrows():
	if index < 10:
		print(row)
	else:
		break

#------------------------------

#filtering
filter = df[(df['Campaign_ID'] == 293787647989)]
print(filter.info())

#summary
print(df['Clicks'].value_counts().head(10))

#ordering
print(df.sort_values('Clicks', ascending=False).head(10))

#------------------------------

#load massive dataset as small chunks

c_size = 10

for gm_chunk in pd.read_csv("train.csv", chunksize=c_size):
	#print(gm_chunk)
	numpy_chunk = gm_chunk.values #to numpy array
	#pd.DataFrame(numpy_chunk) #numpy to pandas restoration
	#print(numpy_chunk.shape," shaped ",type(numpy_chunk))
	print(numpy_chunk)
	break

