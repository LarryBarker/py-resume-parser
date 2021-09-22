# Analyzing Employment Resume Data with Python

## Intro
This program attempts to analyze the impact of resume data on a person's ability to find a job. Do longer resumes, in terms of word count, help find better jobs (higher pay, less time to find)? Does the number of jobs and the length of time matter when it comes to a starting wage?

## Examples
```
print("Reading employment data...")
df = pd.read_csv(file)
print("Creating employment data frame...")
df['arrival_date'] = pd.to_datetime(df['arrival_date'])
df['release_date'] = pd.to_datetime(df['release_date'])
print(df.head(50))
return df

print("Parsing resumes... ")
for i, file in enumerate(files):
    resume = Resume(join(resume_dir, file))
    name = resume.getName()
    for index, row in df.iterrows():
        if row['name'] == name:
            df.loc[index,'word_count'] = resume.getWordCount()
            df.loc[index,'job_count'] = resume.numberOfJobs
            df.loc[index,'avg_job_len'] = resume.lengthOfJobs
            df.loc[index,'months_to_release'] = df.loc[index,'release_date'] - df.loc[index,'arrival_date']
            df.loc[index,'months_to_release'] = round(df.loc[index,'months_to_release'] / np.timedelta64(1, 'M'))
print("Finished parsing resumes...")
```

## Installation

1. Make sure you have NLTK libraries installed from nltk.org
2. Run the Python interpreter and install NLTK:
`import nltk`
`nltk.download()`
A new window should open, showing the NLTK downloader. Click on the file menu and select Change Download Directory. For central installation, set this to:
C:\nltk_data (Windows)
/usr/local/share/nltk_data (Mac)
/usr/share/nltk_data (Unix)
Download ALL packages.
4. Next, install the Python docx module:
`pip install`
`python-docx`
5. Open and run the `analyze.py` file in Spider IDE
6. Enter the file `employed_data.csv` at the first prompt
7. Enter the folder name `resumes` at the second prompt
8. Let the program run and view the output in Spyder console
