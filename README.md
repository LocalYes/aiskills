# aiskills
Semantic skill analysis for job postings, collage courses, and high school courses.

#Components:
- nltk sentence tokinizer to seperate input text into sentences.
- text-embedding-3-large for skill and text embeddings.
- skill filtering algorithm to insure attention to skills.
- ChatGPT feedback loop to filter out FPs and FNs. (and in the case of courses, assess whether some skill is a realistic expectation).
- outputs a ready JSON file.

#Other information
The system is almost fully based on cloud computing, thanks to OpenAI API.

Price and processing latency 
- around 50k words / 1$        (200 job postings)
- around 35k words / 1 hours   (150 job postings) 

