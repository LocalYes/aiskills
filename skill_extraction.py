import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import copy
from collections import Counter
import os

from utils import embed 
from utils import text_segmentation 
from utils import attention_mechanism
from utils import load_skill_embeddings
from utils import text_encoding_flattening

def extract_skills_text_list(text_list, TOP_K_MASKED_SKILLS, TOP_K_SKILLS):
    # Load skill embeddings and names
    base_dir = os.path.dirname(__file__) 
    embed_path = os.path.join(base_dir, "data", "ESCO_embeddings.pkl")
    skill_embeddings, skill_names = load_skill_embeddings.load_skill_embeddings(embed_path)

    # split text into bactches
    batch_word_limit = 5500 # ~8191 tokens, limit for text-embedding-3-large

    text_batches = [[]]
    for text in text_list:
        if sum([len(text_batch.split()) for text_batch in text_batches[-1]]) + len(text.split()) < batch_word_limit:
            text_batches[-1].append(text)
        else:
            text_batches.append([text])

    all_skills_to_return = []
    for text_batch in text_batches:
        """1. Segement the text into sentences and RAKE phrases"""
        # SB - Sentence Based
        # RB - RAKE Based
        SB_text_batch = [text_segmentation.segment_stanza(text) for text in text_batch]
        RB_text_batch = [text_segmentation.segment_RAKE(text, 40) for text in text_batch]

        """2. Embed the segments"""
        # Flatten
        SB_to_embed_key, SB_to_embed = text_encoding_flattening.flatten_batch(SB_text_batch)
        RB_to_embed_key, RB_to_embed = text_encoding_flattening.flatten_batch(RB_text_batch)

        # Embed
        SB_embed_encode = embed.openai_embed(SB_to_embed)
        RB_embed_encode = embed.openai_embed(RB_to_embed)

        # Unflatten
        SB_embed = text_encoding_flattening.unflatten_batch(SB_embed_encode, SB_to_embed_key)
        RB_embed = text_encoding_flattening.unflatten_batch(RB_embed_encode, RB_to_embed_key)

        # Loop over each job
        for job_i in range(0, len(text_batch)):
            SB_embed_job = np.array(SB_embed[job_i])
            RB_embed_job = np.array(RB_embed[job_i])

            """3. Cosine similarity"""
            SB_sim = cosine_similarity(SB_embed_job, skill_embeddings)
            RB_sim = cosine_similarity(RB_embed_job, skill_embeddings)

            """4. Apply custom attention mechanism"""
            SB_skill_scores = attention_mechanism.top_k_masked_mean_pooling(TOP_K_MASKED_SKILLS, SB_sim)
            RB_skill_scores = attention_mechanism.top_k_masked_mean_pooling(TOP_K_MASKED_SKILLS, RB_sim)

            skill_scores = 0.5 * SB_skill_scores + 0.5 * RB_skill_scores

            # Retrieve top skills and scores
            top_idxs = np.argpartition(skill_scores, -TOP_K_SKILLS)[-TOP_K_SKILLS:]
            top_idxs = top_idxs[np.argsort(skill_scores[top_idxs])[::-1]]

            top_skills_with_scores = [
                (skill_names[j].replace("\u00a0", " "), float(skill_scores[j]))
                for j in top_idxs
            ]

            all_skills_to_return.append(top_skills_with_scores)
    
    return all_skills_to_return


base_dir = os.path.dirname(__file__)
json_path = os.path.join(base_dir, "data", "ESCO_hierarchies.json")

# Pre load skill hierarchy
with open(json_path, "r") as file:
    hierarchy = json.load(file)

def get_skills_clusters(skills_list, cluster_branch_index, top_clusters_used=-1):
    skills_list = copy.deepcopy(skills_list)
    cluster_score_count = []
    for branch in hierarchy:
        branch = branch["hierarchy"]
        if len(branch) >= cluster_branch_index+1 and branch[-1] in skills_list:
            skills_list.remove(branch[-1])
            cluster_score_count.append(branch[cluster_branch_index])
    
    cluster_score_count = Counter(cluster_score_count)

    if top_clusters_used == -1: return dict(cluster_score_count)

    top_clusters = cluster_score_count.most_common(top_clusters_used)
    return dict(top_clusters)

def retrieve_skills_from_clusters(cluster_list, cluster_branch_index):
    skills_from_clusters = []

    for branch in hierarchy:
        branch = branch["hierarchy"]
        if len(branch) >= cluster_branch_index+1 and branch[cluster_branch_index] in cluster_list:
            skills_from_clusters.append(branch[-1])

    return skills_from_clusters


