import json
from skill_extraction import extract_skills_text_list, get_skills_clusters, retrieve_skills_from_clusters
from skill_filtering import chatgpt_skills_filtering
from utils import truncate_skills
import os

base_dir = os.path.dirname(__file__)
example_data_path = os.path.join(base_dir, "example_data.json")

with open(example_data_path, "r", encoding="utf-8") as file: 
    example_data = json.load(file)


"""Get top skill matches"""
TOP_K_MASKED_SKILLS = 13000
TOP_K_SKILLS = 80

top_skills_data = extract_skills_text_list(example_data, TOP_K_MASKED_SKILLS, TOP_K_SKILLS)
"""Get top skill matches"""



"""Expand skills from the clusters"""
top_skills_data = [[skill[0] for skill in top_skills] for top_skills in top_skills_data]
TOP_OG_SKILLS = 40

CLUSTER_BRANCH_INDEX = 4 # (1,2,3,4) The granularity of the skills clusters. (eg. 1 might be "management, CS, math", and 2 might be "management of medical staff", "Java", "Statistical analysis") 4 is playing it safe
TOP_CLUSTERS_USED = 8

for all_skills in top_skills_data:
    top_og_skills = truncate_skills.get_top_x_skills(all_skills, TOP_OG_SKILLS)
    clusters = get_skills_clusters(all_skills, CLUSTER_BRANCH_INDEX, TOP_CLUSTERS_USED)
    cluster_skills = retrieve_skills_from_clusters(clusters, CLUSTER_BRANCH_INDEX)
    print(f"skill added from cluster_skills: {len(cluster_skills)}")

    top_skills = list(set(top_og_skills + cluster_skills))
    print(f"\ntop_skills - (not_filtered)\n{top_skills}\n")
"""Expand skills from the clusters"""



"""Filter skills using ChatGPT"""
skills_filtered = chatgpt_skills_filtering(example_data, top_og_skills, "gpt-4.1-mini")

print(f"\ntop_skills - (filtered)\n{skills_filtered}\n")
"""Filter skills using ChatGPT"""

