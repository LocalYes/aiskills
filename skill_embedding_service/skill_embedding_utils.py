import numpy as np
import pickle 
import pandas as pd

def csv_skills_data_load(path, goal_column, check_statement=["Skills", "ESCO"]):
    skills_list = []
    df = pd.read_csv(path)

    for i, row in df.iterrows():
        if check_statement: 
            if check_statement[1] in row[check_statement[0]]:
                skills_list.append(row[goal_column])
        else:
            skills_list.append(row[goal_column])
        
    return skills_list


def save_skill_embeddings(skill_embeddings, skill_names, path):
    if not isinstance(skill_embeddings, np.ndarray):
        raise TypeError("skill_embeddings must be a numpy array")

    all_skills = []
    for emb, name in zip(skill_embeddings, skill_names):
        skill_dict = {
            "CentriodEmbedding": emb,
            "SkillData": [name]
        }
        all_skills.append(skill_dict)

    with open(path, "wb") as f:
        pickle.dump(all_skills, f)