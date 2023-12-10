import pickle

# TODO: Preparing Model
with open("model/le_edu.pkl", "rb") as edu:
    le_edu = pickle.load(edu)

with open("model/le_gender.pkl", "rb") as gender:
    le_gender = pickle.load(gender)

with open("model/le_job.pkl", "rb") as job:
    le_job = pickle.load(job)

def preprocessor(dataframe):
    data = dataframe.copy()
    data["Gender"] = le_gender.transform(data["Gender"])
    data["Education Level"] = le_edu.transform(data["Education Level"])
    data["Job Title"] = le_job.transform(data["Job Title"])
    
    return data