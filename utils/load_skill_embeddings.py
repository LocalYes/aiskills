import pickle 
import numpy as np
import os

def load_skill_embeddings(path):
    skill_embeddings = []
    skill_names = []
    with open(path, "rb") as f:
        all_skills = pickle.load(f)

    for i, skill in enumerate(all_skills):
        skill_embeddings.append(skill["CentriodEmbedding"])
        skill_names.append(skill["SkillData"][0])

    return np.array(skill_embeddings), skill_names