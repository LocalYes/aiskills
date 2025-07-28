# aiskills
Semantic skill analysis for job postings, college courses, and high school courses.

cheap, accurate, customizable

## Process overview

- Uses Stanza and RAKE to split input text into sentences.
- Embeds both skills `(ESCO 13800)` and text with `text-embedding-3-large`.
- cosine similarity
- Runs a custom skill filtering algorithm.
- Includes an optional ChatGPT filtering step to reduce false positives/negatives.

The repository includes main.py, which roughly demonstrates how this process can be done. 
Output example is at the end.

## Optional features

- **Enhance skill coverage with ESCO clusters:**
  1. Extract skills from your text with `extract_skills_text_list` (ideally with `TOP_K_SKILLS` high).
  2. Use `truncated_skills` to create two sets: a full set, and a shorter, "real" set.
  3. `get_skills_clusters` can identify popular skill clusters in the large set.
  4. Use `retrieve_skills_from_clusters` to extend the "real" set with added skills,
  6. Continue as normal, it is recommended to use ChatGPT filtering in this case. 

This setup keeps you close to the important skills from the embeddings and brings in all the skills from popular clusters. If you use ChatGPT for review, you’re much less likely to miss something important.

- **Adjust skill granularity:**
  1. Smaller `TOP_K_MASKED_SKILLS` - `(100 - 13000)` values: lets through rarer but high-confidence skills, which appear in only a few sentences.
  2. Larger values: lets through more frequent skills, even if confidence is moderate.
  3. Note: If your data is noisy, setting `TOP_K_MASKED_SKILLS` too low will hurt results.

- **Adjust skill to a requirement (rationale):**
  1. `chatgpt_skills_filtering` includes two parameters `include_rationale_step (T/F)` and `rationale_statement (String)`
  2. Enabling the first and setting the second to something such as "skills must be on a level reasonable for a junior/senior highschool student to know." would make the model go over each skill an rationalize if it should be present.

## Other notes
- Runs almost entirely in the cloud using the OpenAI API. No GPU requirements.
- **Cost and speed:**
  - About `200,000` words per $1        **//That is with all ChatGPT filtering enabled**
  - About `160,000` words per hour
 
  - About `750,000` words per $0.1     **//That is only using embeddings, consequently about 5 times faster.**
  - About `600,000` words per hour
    
- Note, it's a developement version, few things are still rough.

## Example

Accounting Technician I

link to posting: https://www.governmentjobs.com/careers/northcarolina/jobs/5016399/accounting-technician-i?page=2&pagetype=jobOpportunitiesJobs

To demonstrate model's abilities, the description was copied together with noise (eg. "please contact Lyndsey Schrier at (910) 944-2359").


Output:

['accounting_methods', 'trust_fund_accounting', 'offender_welfare_accounting', 'fiscal_policy_compliance', 'purchasing_procedures', 'accounts_payable', 'data_analysis', 'problem_solving', 'microsoft_excel', 'microsoft_word', 'spreadsheet_management', 'document_log_maintenance', 'office_equipment_operation']
