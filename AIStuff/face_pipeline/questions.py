"""
@author: Xiang Lan
@file: questions.py
@time: 2023/1/3 15:06
@desc: include the questions for active liveness detection
"""

def question_bank(index):
    questions = [
                "smile",
                "surprise",
                "blink eyes",
                "angry",
                "turn face right",
                "turn face left",
                "turn face up",
                "turn face down",
    ]
    return questions[index]


def challenge_result(question, out_model,blinks_up):
    if question == "smile":
        if len(out_model["emotion"]) == 0:
            challenge = "fail"
        elif out_model["emotion"][0] == "smile":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "surprise":
        if len(out_model["emotion"]) == 0:
            challenge = "fail"
        elif out_model["emotion"][0] == "surprise":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "angry":
        if len(out_model["emotion"]) == 0:
            challenge = "fail"
        elif out_model["emotion"][0] == "angry":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "turn face right":
        if len(out_model["orientation"]) == 0:
            challenge = "fail"
        elif out_model["orientation"][0] == "right":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "turn face left":
        if len(out_model["orientation"]) == 0:
            challenge = "fail"
        elif out_model["orientation"][0] == "left":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "turn face up":
        if len(out_model["orientation"]) == 0:
            challenge = "fail"
        elif out_model["orientation"][0] == "up":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "turn face down":
        if len(out_model["orientation"]) == 0:
            challenge = "fail"
        elif out_model["orientation"][0] == "down":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "blink eyes":
        if blinks_up == 1:
            challenge = "pass"
        else:
            challenge = "fail"

    return challenge
