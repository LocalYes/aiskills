import pickle 
import numpy as np
import os

base_dir = os.path.dirname(__file__) 
parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
embed_path = os.path.join(parent_dir, "data", "ESCO_embeddings.pkl")

def load_skill_embeddings(path=embed_path):
    skill_embeddings = []
    skill_names = []
    with open(path, "rb") as f:
        all_skills = pickle.load(f)

    for i, skill in enumerate(all_skills):
        skill_embeddings.append(skill["CentriodEmbedding"])
        skill_names.append(skill["SkillData"][0])

    return np.array(skill_embeddings), skill_names