import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from skill_embedding_utils import csv_skills_data_load, save_skill_embeddings
from aiskills.utils.embed import openai_embed



base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "raw_skills", "test.csv")

skills_list = csv_skills_data_load(csv_path, "Skill", None)

print("SKILLS EXTRACTED: ", len(skills_list))
print("Word count: ", len((" ".join(skills_list)).split(" ")))

# If the word count exceeds ~5000 words, do a split
embedded_skills = np.concatenate([
    openai_embed(skills_list[:1000]),
    openai_embed(skills_list[1000:])
], axis=0)

print("SHAPE OF EMBEDDINGS: ", embedded_skills.shape)

save_skill_embeddings(embedded_skills, skills_list, r"C:\Users\artio\OneDrive\Desktop\aiskills\data\ESCO_embeddings_small.pkl")


