# aiskills

Semantic skill analysis for job postings, college courses, and high school courses.

## Process overview

- Uses stanza and RAKE to split input text into sentences.
- Embeds both skills and text with `text-embedding-3-large`.
- Runs a custom skill filtering algorithm—returns top matching skills using cosine similarity.
- Includes a ChatGPT feedback loop to reduce false positives/negatives. For courses, it can also check if certain skills are realistic.
- Outputs a ready-to-use JSON file.

## Optional features

- **Enhance skill coverage with ESCO clusters:**
  1. Extract skills from your text with `extract_skills_text_list` (ideally with `TOP_K_SKILLS=300`).
  2. Use `get_truncated_skills` to create two sets: a full set (300 skills, used for clusters), and a shorter, "real" set (20-30 recommended).
  3. `populate_skills_using_clusters` can identify popular skill clusters in the large set.
  4. Add the relevant cluster skills to your real set.
  5. Continue as normal.

This setup keeps you close to the important skills from the embeddings and brings in all the skills from popular clusters. If you use ChatGPT for review, you’re much less likely to miss something important.

- **Adjust skill granularity:**
  1. Smaller `TOP_K_MASKED_SKILLS` values (300–4,000): lets through rarer but high-confidence skills (appearing in only a few sentences).
  2. Larger values (11,000–13,000): lets through more frequent skills, even if confidence is moderate.
  3. Note: If your data is noisy, setting `TOP_K_MASKED_SKILLS` too low will hurt results.

## Other notes

- Runs almost entirely in the cloud using the OpenAI API. No GPU required.
- **Cost and speed:**
  - About 50,000 words per $1 (roughly 200 job postings)
  - About 35,000 words per hour (roughly 150 job postings)
