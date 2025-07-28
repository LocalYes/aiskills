def get_top_x_skills(skills, x):
    # Insure that the the skills are sorted.

    top_x_skills = []
    for skill in skills:
        top_x_skills.append(skill)
        x -= 1
        if x <= 0: 
            return top_x_skills

