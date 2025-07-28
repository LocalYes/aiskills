import pickle 
import numpy as np

def load_skill_embeddings(path=r"C:\Users\artio\OneDrive\Desktop\2_semantic-course-to-skill\saved_skill_embed\stage_1_UN_skills_1.pkl"):
    skill_embeddings = []
    skill_names = []
    with open(path, "rb") as f:
        all_skills = pickle.load(f)

    for i, skill in enumerate(all_skills):
        skill_embeddings.append(skill["CentriodEmbedding"])
        skill_names.append(skill["SkillData"][0])

    return np.array(skill_embeddings), skill_names