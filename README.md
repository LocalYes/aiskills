# aiskills                                                    
Semantic skill analysis for job postings, collage courses, and high school courses.

# Rough process description:
- stanza and RAKE to segment input text into sentences.
- text-embedding-3-large for skill and text embeddings.
- custom skill filtering algorithm, to find top matching skills after cosine similarity
- ChatGPT feedback loop to filter out FPs and FNs. (and in the case of courses, assess whether some skill is a realistic expectation).
- outputs a ready JSON file.

# Optional features:
- Enahnce the skill coverage using clusters from the ESCO skill hierarchy.
  1. Get the skills for your text using *extract_skills_text_list* preferebly with *TOP_K_SKILLS=300*
  2. Use the *get_truncated_skills* to get two versions of your skills, the original one with 300 (used for clusters), and the real skills from *get_truncated_skills*, preferebly 20-30.
  3. Use *populate_skills_using_clusters* to find most popular skill clusters from the 300 skills.
  4. Extend the real skills with the clusters skills.
  5. continue...

(This process allows to have the most important skills recieved from the embeddings, and include ALL of the skills from most popular skill clusters, which, if you are using ChatGPT, is likely to gaurenty for no missed skills).

- Adjust the skill granularity, by changing the *TOP_K_MASKED_SKILLS*
  1. By making *TOP_K_MASKED_SKILLS* smaller you allow for skills which appear only in a few sentences, but with high confidence, to pass. Set value to (300 - 4,000)
  2. By making *TOP_K_MASKED_SKILLS* bigger you allow for skills which appear a lot in many sentence to pass. Set value to (11,000 - 13,000)

(**IMPORTANT:** if your data is noise, a small *TOP_K_MASKED_SKILLS* will lead to poor performance.)

# Other information
The system is almost fully based on cloud computing, thanks to OpenAI API. No GPU requirements

Price and processing latency 
- around 50k words / 1$        (200 job postings)
- around 35k words / 1 hours   (150 job postings) 

