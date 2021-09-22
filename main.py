"""
Larry Barker
CS 521
8/17/18
Final Project
Performs cluster analysis on employment and resume data to determine
how resume variables affect employment metrics (starting wage and time to find
a job). Contains methods to make the primary data frame and parse the resume
files in a specified directory. Cleans the data and assigns cluster variables
to be split into test and train sets. 
"""
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

from resume import Resume

def make_dataframe(file):
    """
    Description: create a dataframe from the csv file and convert the dates to
    date time objects so they can be manipulated later
    :input param: file = csv file to create a dataframe from
    :return: df = pandas dataframe object
    """
    print("Reading employment data...")
    df = pd.read_csv(file)
    print("Creating employment data frame...")
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    df['release_date'] = pd.to_datetime(df['release_date'])
    print(df.head(50))
    return df

def parse_resumes(files, directory):
    """
    Description: return a Document object for the resume
    :input param: files = list of files in a directory
    :input param: directory = directory where resumes are found
    :return: df = merged data frame with resume information
    """
    print("Parsing resumes... ")
    for i, file in enumerate(files):
        resume = Resume(join(directory, file))
        name = resume.getName()
        for index, row in df.iterrows():
            if row['name'] == name:
                df.loc[index,'word_count'] = resume.getWordCount()
                df.loc[index,'job_count'] = resume.numberOfJobs
                df.loc[index,'avg_job_len'] = resume.lengthOfJobs
                df.loc[index,'months_to_release'] = df.loc[index,'release_date'] - df.loc[index,'arrival_date']
                df.loc[index,'months_to_release'] = round(df.loc[index,'months_to_release'] / np.timedelta64(1, 'M'))
    print("Finished parsing resumes...")
    #print("Saving employment data...")
    return df

# get employment data file from input
while True:
    file_name = input('Enter name of CSV file with employment data: ')
    try:
        df = make_dataframe(file_name)
        break
    except IOError:
        print("Could not read file:", file_name)
        
# check resume directory for files
while True:
    resume_dir = input("Enter the name of directory with resumes: ")
    try:
        onlyfiles = [f for f in listdir(resume_dir) if isfile(join(resume_dir, f))]
        break
    except Exception as e:
        print(e)

data = parse_resumes(onlyfiles, resume_dir)

''' BEGIN cluster analysis '''

print("Beginning cluster analysis...")

# data management

data.columns = map(str.upper, data.columns)

data_clean = data.dropna()

# subset clustering variables
cluster = data_clean[['LEAD_TIME','WORD_COUNT','JOB_COUNT','AVG_JOB_LEN','MONTHS_TO_RELEASE']]

print(cluster.describe())

# standardize clustering variables to have mean=0 and sd=1
clustervar = cluster.copy()

clustervar['LEAD_TIME'] = preprocessing.scale(clustervar['LEAD_TIME'].astype('float64'))
#clustervar['START_WAGE'] = preprocessing.scale(clustervar['START_WAGE'].astype('float64'))
clustervar['WORD_COUNT'] = preprocessing.scale(clustervar['WORD_COUNT'].astype('float64'))
clustervar['JOB_COUNT'] = preprocessing.scale(clustervar['JOB_COUNT'].astype('float64'))
clustervar['AVG_JOB_LEN'] = preprocessing.scale(clustervar['AVG_JOB_LEN'].astype('float64'))
clustervar['MONTHS_TO_RELEASE'] = preprocessing.scale(clustervar['MONTHS_TO_RELEASE'].astype('float64'))

# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) / clus_train.shape[0])

# plot average distance from observations from the cluster centroid
plt.plot(clusters, meandist)
plt.xlabel('Number of clusers')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.show()

print("Performing cluster analysis with 3 clusters...")

model3 = KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign = model3.predict(clus_train)

# plot clusters
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_)
plt.xlabel('Cannonical variable 1')
plt.ylabel('Cannonical variable 2')
plt.title('Scatterplot of Cannonical Variables for 3 Clusters')
plt.show()

''' END cluster analysis '''

''' BEGIN multiple steps to merge cluster assignment with clustering variables
to examine cluster variable means by cluster '''
# create unique id variable from the index
# for the cluster training data to merge the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)

# create a list that has the new index variable
cluslist=list(clus_train['index'])

# create a list of cluster assignments
labels=list(model3.labels_)

# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist

# convert new list dict to a df
newclus=DataFrame.from_dict(newlist, orient='index')
newclus

# do the same for the cluster assignment variable
# rename the cluster assignment column
newclus.columns = ['cluster']

#create a unique id var from the index for the
# cluster assignment df to merge w/ cluster training data
newclus.reset_index(level=0, inplace=True)

# merge the cluster assignment df w/ the cluster training variable df
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
print(merged_train.head(n=50))

# cluster frequencies
merged_train.cluster.value_counts()

''' END multiple steps to merge cluster assignment with clustering variables
to examine cluster variable means by cluster '''

''' FINALLY calculate clustering variable means by cluster '''
clustergrp = merged_train.groupby('cluster').mean()
print("Clustering variable means by cluster")
print(clustergrp)

# validate clusters in training data by examining cluster differences in START_WAGE
# using ANOVA.
# first need to merge START_WAGE with clustering variables and cluster assignment data
start_wage_data=data_clean['START_WAGE']

# split START_WAGE data into train and test sets
start_wage_train, start_wage_test = train_test_split(start_wage_data, test_size=.3, random_state=123)
start_wage_train1=pd.DataFrame(start_wage_train)
start_wage_train1.reset_index(level=0,inplace=True)
merged_train_all=pd.merge(start_wage_train1,merged_train,on='index')
sub1=merged_train_all[['START_WAGE','cluster']].dropna()

startwagemod = smf.ols(formula='START_WAGE ~ C(cluster)',data=sub1).fit()
print(startwagemod.summary())

print('means for START_WAGE by cluster')
m1=sub1.groupby('cluster').mean()
print(m1)

print('standard deviations for START_WAGE by cluster')
m2=sub1.groupby('cluster').std()
print(m2)

mc1 = multi.MultiComparison(sub1['START_WAGE'],sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())