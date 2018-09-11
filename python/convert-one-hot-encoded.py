import pandas as pd
import time
#------------------------------------
#loading dataset
begin = time.time()
df = pd.read_csv("adult.data"
	, names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "earning"])
print("dataset loaded in ",time.time()-begin," seconds")

#------------------------------------
rows = df.shape[0] - 1
columns = df.shape[1]

"""
#dataset summary
for i in range(0, columns):			
	if df[df.columns[i]].dtypes != "int64":
		print(df.columns[i],": ",df[df.columns[i]].unique()," (",len(df[df.columns[i]].unique())," classes)")
	else:
		print(df.columns[i])
"""

#------------------------------------

f = open('one-hot-encoded.txt', 'w')

#dump header
header = ""
for i in range(0, columns):
	
	if i == 0:
		seperator = ""
	else:
		seperator = ","	
	
	if df[df.columns[i]].dtypes != "int64":		
		for k in range(0, len(df[df.columns[i]].unique())):
			header += seperator + df[df.columns[i]].unique()[k]	
	else:
		header += seperator + df.columns[i]


header += "\n"	
#print(header)
f.write(header)

#------------------------------------

#iterate on rows
for index, row in df.iterrows():
	new_line = ""
	#iterate on columns
	for i in range(0, columns):
	
		if i == 0:
			seperator = ""
		else:
			seperator = ","
			
		column_name = df.columns[i]
		if df[df.columns[i]].dtypes == "int64":
			new_line = new_line + seperator + str(row[column_name])
		else: #class
			num_hot_encoded_classes = len(df[df.columns[i]].unique())
			for k in range(0, num_hot_encoded_classes):
				if df[df.columns[i]].unique()[k] == row[column_name]:
					new_line = new_line + seperator + "1"
				else:
					new_line = new_line + seperator + "0"
	
	new_line += "\n"
	#print(new_line)			
	f.write(new_line)

#------------------------------------

f.close()
print("converted to one-hot-encoded dataset in ",time.time()-begin," seconds")