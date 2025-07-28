import json
import time
from openai import OpenAI


# Initialize the OpenAI client
client = OpenAI(api_key=open(r"C:\Users\artio\OneDrive\Desktop\aiskills\OPENAI_KEY.txt").read().strip())

def chatgpt_send_messages(messages, model):
    analytical_response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return analytical_response.choices[0].message.content

def chatgpt_send_messages_json(messages, json_schema, model):
    json_response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "responce_retrieval",
                "strict": True,
                "schema": json_schema
            }
        }
    )

    json_response_content = json_response.choices[0].message.content
    return json.loads(json_response_content)


def chatgpt_skills_filtering(text_list, skills_list, gpt_model="gpt-4.1-mini", include_rationale_step=False, rationale_statement="skills must be on a level reasonable for a junior/senior highschool student to know."):
    # text_list and skills_list must be parallel lists.
    
    filtered_skills_list = []
    time_update = time.time()

    for i, (text, skills) in enumerate(zip(text_list, skills_list)):
        print(f"\nFILTERING TEXT   ( {i+1} / {len(text_list)} )  ~  {round(time.time() - time_update, 3)}s")
        time_update = time.time()

        print("1. analytical step")

        analytical_messages = [
            {"role": "system", "content": '''
                You are a skills evaluator tasked with determining the alignment between a set of estimated skills and a provided job posting or course description. Your goal is to classify and refine the skill list based on how well the skills match the responsibilities and expectations described in the text. Follow the instructions below:

                1. Highly Relevant Skills
                List the skills that are clearly aligned with the job or course. These skills should either be explicitly mentioned or strongly implied.

                2. Possibly Relevant but Less Central
                List the skills that might apply indirectly, support peripheral functions, or are related but not essential.

                3. Irrelevant or Misaligned Skills
                List the skills that do not match the description in domain, level, or responsibility.

                4. Final Refined Skill List
                Based only on section 1 (and a few justified entries from section 2 if applicable), generate a final Python-style list of skills appropriate for the role.

                The final list should represent only the most relevant and context-appropriate skills for the job or course. Format it like this:

                python
                Copy
                Edit
                [
                    "skill_1",
                    "skill_2",
                    ...
                ]
                Avoid emojis or unnecessary commentary.
            '''},
            {"role": "user", "content": f'''
                TEXT:
                {text}

                PREDICTED SKILLS:
                {skills}
            '''}
        ]

        analytical_responce = chatgpt_send_messages(analytical_messages, gpt_model)

        rational_responce = None
        if include_rationale_step:
            print("2. rational step")
            analytical_skills_estimated = analytical_responce.split("4.")[-1]

            rational_messages = [
                {"role": "system", "content": f'''
                    You are a second-stage evaluator. You have received a list of estimated skills selected as suitable for a job or course, based on an initial match with a given description. Now, your job is to **critically reflect on whether each of these skills is realistic or appropriate** for the individual expected to take this role or course, based on the following rationale:

                    **Rationale:** {rationale_statement}

                    Instructions:
                    1. Evaluate each skill in the refined list and ask: "Is this skill truly realistic for someone at this level or context, according to the rationale?"
                    2. Remove any skills that are unrealistic or too advanced, even if they align with the job or course content.
                    3. Keep skills that are plausibly learnable, teachable, or applicable within the given constraint.

                    Your final output should be a Python-style list of only those skills from the prior refined list that are realistic and appropriate given the rationale.

                    Format strictly like this:

                    ```python
                    [
                        "skill_1",
                        "skill_2",
                        ...
                    ]
                    Do not add any other explanation or commentary.
                '''},
                {"role": "user", "content": f'''
                    Here is the previously refined skill list to reassess based on the rationale.

                    Refined Skill List:
                    {analytical_skills_estimated}
                '''}
            ]

            rational_responce = chatgpt_send_messages(rational_messages, gpt_model)
        

        print("3. jsonification step")

        json_messages = [
            {"role": "system", "content": """
                You are a JSON transformation assistant.
                Your task is to read the user's input, which contains a list of final skills (typically formatted in Python-style or plain text), and convert it into a valid, JSON structure, obeying the json schema.

                Objective:
                1. Extract the final list of skills from user input, and output the list of skills acording to the json schema.
                2. Copy the skills letter to letter, no additional corrections or capitalization. 
            """},
            {"role": "user", "content": f"""
                {rational_responce or analytical_responce}
            """}
        ]

        json_schema = {
            "type": "object",
            "properties": {
                "skills": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["skills"],
            "additionalProperties": False
        }

        filtered_skills = chatgpt_send_messages_json(json_messages, json_schema, "gpt-4.1-nano")
        filtered_skills_list.append(filtered_skills["skills"])


    return filtered_skills_list
