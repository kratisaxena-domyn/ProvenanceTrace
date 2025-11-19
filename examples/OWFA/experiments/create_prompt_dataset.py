import json, os
import random


# Domain-agnostic Wikipedia-aligned prompt generator
import json, os
import random

save_folder = "experiment_data/prompt_datasets/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# Wikipedia topic categories and subcategories with topic types
WIKI_CATEGORIES = {
    "History": {
        "Ancient": [
            {"name": "Roman Empire", "type": "event"},
            {"name": "Egypt", "type": "place"},
            {"name": "Mesopotamia", "type": "place"}
        ],
        "Modern": [
            {"name": "World War II", "type": "event"},
            {"name": "Cold War", "type": "event"},
            {"name": "Industrial Revolution", "type": "event"}
        ]
    },
    "Geography": {
        "Countries": [
            {"name": "France", "type": "place"},
            {"name": "Japan", "type": "place"},
            {"name": "Brazil", "type": "place"}
        ],
        "Landforms": [
            {"name": "Mount Everest", "type": "place"},
            {"name": "Sahara Desert", "type": "place"},
            {"name": "Amazon River", "type": "place"}
        ]
    },
    "Arts": {
        "Literature": [
            {"name": "Shakespeare", "type": "person"},
            {"name": "Harry Potter", "type": "work"},
            {"name": "Moby-Dick", "type": "work"}
        ],
        "Music": [
            {"name": "Beethoven", "type": "person"},
            {"name": "Jazz", "type": "genre"},
            {"name": "The Beatles", "type": "group"}
        ],
        "Painting": [
            {"name": "Mona Lisa", "type": "work"},
            {"name": "Picasso", "type": "person"},
            {"name": "Impressionism", "type": "genre"}
        ]
    },
    "People": {
        "Historical Figures": [
            {"name": "Albert Einstein", "type": "people"},
            {"name": "Cleopatra", "type": "people"},
            {"name": "Genghis Khan", "type": "people"}
        ],
        "Scientists": [
            {"name": "Marie Curie", "type": "people"},
            {"name": "Isaac Newton", "type": "people"},
            {"name": "Charles Darwin", "type": "people"}
        ],
        "Celebrities": [
            {"name": "Oprah Winfrey", "type": "people"},
            {"name": "Leonardo DiCaprio", "type": "people"},
            {"name": "Beyoncé", "type": "people"}
        ]
    },
    "Organizations": {
        "Companies": [
            {"name": "Apple Inc.", "type": "org"},
            {"name": "Google", "type": "org"},
            {"name": "Microsoft", "type": "org"}
        ],
        "Non-Profits": [
            {"name": "Red Cross", "type": "org"},
            {"name": "UNICEF", "type": "org"},
            {"name": "WWF", "type": "org"}
        ],
        "Government": [
            {"name": "United Nations", "type": "org"},
            {"name": "NATO", "type": "org"},
            {"name": "European Union", "type": "org"}
        ]
    }

}

ADVANCED_TEMPLATES = {
    "concept": [
        "Compare {topic} with another major concept in its field.",
        "How would the world be different if {topic} did not exist?",
        "What are common misconceptions about {topic}?",
        "How does {topic} relate to current technological advancements?"
    ],
    "object": [
        "How has our understanding of {topic} changed over time?",
        "What would happen if {topic} ceased to function?",
        "Compare the role of {topic} in two different biological systems."
    ],
    "process": [
        "What factors can disrupt the process of {topic}?",
        "How does {topic} interact with other scientific processes?",
        "Describe a scenario where {topic} fails to occur."
    ],
    "artifact": [
        "How has the invention of {topic} influenced society?",
        "What are the kinds of {topic}?",
        "Tell me about the history of {topic} and its evolution over time."
    ],
    "event": [
        "What were the unintended consequences of {topic}?",
        "How might history have changed if {topic} had not occurred?",
        "Tell me the history of {topic}."
    ],
    "place": [
        "How has the geography of {topic} influenced its history?",
        "What challenges does {topic} face today?",
        "Tell me some unique facts about {topic}."
    ],
    "person": [
        "What controversies surrounded {topic}?",
        "How might history have changed without {topic}?",
        "What are the achievements and failures of {topic}?"
    ],
    "work": [
        "How has {topic} influenced later works?",
        "What criticisms have been made about {topic}?",
        "Tell me some interesting facts about {topic}."
    ],
    "genre": [
        "How has the genre {topic} evolved over time?",
        "What are the challenges in defining {topic}?",
        "Who are some immininent figures in the genre of {topic}?"
    ],
    "group": [
        "How did {topic} change the landscape of their field?",
        "What internal conflicts affected {topic}?",
        "What are some famous achievements of {topic}?"
    ],
    "people": [
        "Tell me about the early life and family of {topic}?",
        "What are some notable works of {topic}?",
        "What awards and recognitions has {topic} received?"
    ],
    "org": [
        "What is the mission and vision of {topic}?",
        "Who are some prominent figures associated with {topic}?",
        "What are some major milestones in the history of {topic}?"
    ]
}


# Context-aware prompt templates by topic type
PROMPT_TEMPLATES = {
    "concept": [
        "What is {topic}?",
        "Explain the concept of {topic}.",
        "Why is {topic} important in its field?",
        "What are the main principles of {topic}?"
    ],
    "object": [
        "What is a {topic}?",
        "Describe the structure and function of {topic}.",
        "Where can {topic} be found?"
    ],
    "process": [
        "What is the process of {topic}?",
        "How does {topic} occur in nature?",
        "Why is {topic} significant in science?"
    ],
    "artifact": [
        "What is {topic} and how is it used?",
        "Describe the development and impact of {topic}.",
        "What are the main features of {topic}?"
    ],
    "event": [
        "What was the significance of {topic}?",
        "Summarize the main events of {topic}.",
        "How did {topic} change history?"
    ],
    "place": [
        "Where is {topic} located?",
        "Describe the geography and importance of {topic}.",
        "What makes {topic} unique?"
    ],
    "person": [
        "Who was {topic}?",
        "What are the major achievements of {topic}?",
        "Why is {topic} famous?"
    ],
    "work": [
        "What is {topic} about?",
        "Who created {topic} and when?",
        "Why is {topic} considered important?"
    ],
    "genre": [
        "What defines the genre {topic}?",
        "What are the key characteristics of {topic}?",
        "Name some famous examples of {topic}."
    ],
    "group": [
        "Who are {topic}?",
        "What are the major accomplishments of {topic}?",
        "Why are {topic} influential in their field?"
    ]
}



FACT_CHECK_TEMPLATES = {
    "event": [
        "In what year did {topic} begin?",
        "In what year did {topic} end?",
        "Name one immediate consequence of {topic}.",
        "Name a principal participant in {topic}.",
        "Name one key leader during {topic}.",
        "Name one major battle or turning point in {topic}.",
        "Name one treaty or agreement linked to {topic}.",
        "Name one long-term outcome of {topic}."
    ],
    "place": [
        "On which continent is {topic} located?",
        "Name one bordering country or region of {topic}.",
        "What is one notable geographical feature of {topic}?",
        "Name one climate characteristic of {topic}.",
        "Name one natural resource found in {topic}.",
        "Name one protected area or landmark within {topic}."
    ],
    "person": [
        "In what year was {topic} born?",
        "In what field is {topic} best known?",
        "Name one major achievement of {topic}.",
        "In what year did {topic} die?",
        "Name one award received by {topic}.",
        "Name one institution associated with {topic}."
    ],
    "people": [
        "In what year was {topic} born?",
        "What nationality is {topic}?",
        "Name one notable work of {topic}.",
        "In what year did {topic} die?",
        "Name one award received by {topic}.",
        "Name one occupation of {topic}."
    ],
    "work": [
        "In what year was {topic} first released or published?",
        "Who is the creator of {topic}?",
        "Name one major theme of {topic}.",
        "Name one principal character in {topic}.",
        "Name one award received by {topic}.",
        "Name one genre classification of {topic}."
    ],
    "genre": [
        "In what decade did {topic} emerge prominently?",
        "Name one influential figure associated with {topic}.",
        "Name one defining characteristic of {topic}.",
        "Name one seminal work in {topic}.",
        "Name one subgenre related to {topic}.",
        "Name one historical origin influencing {topic}."
    ],
    "group": [
        "In what year did {topic} form?",
        "Name one landmark achievement of {topic}.",
        "Name one original member of {topic}.",
        "Name one widely known release by {topic}.",
        "Name one award received by {topic}.",
        "In what year did {topic} disband or peak?"
    ],
    "org": [
        "In what year was {topic} founded?",
        "Where is {topic} headquartered?",
        "Name one core mission focus of {topic}.",
        "Name one major initiative of {topic}.",
        "Name one key leader of {topic}.",
        "Name one subsidiary or branch of {topic}."
    ],
    "artifact": [
        "In what year was {topic} first introduced?",
        "Name one key use-case of {topic}.",
        "Who is credited with creating {topic}?",
        "Name one material commonly used in {topic}.",
        "Name one industry transformed by {topic}.",
        "Name one predecessor or earlier version of {topic}."
    ],
    "process": [
        "Name one necessary input for {topic}.",
        "Name one common outcome of {topic}.",
        "Name one factor that influences {topic}.",
        "Name one intermediate stage of {topic}.",
        "Name one application area of {topic}.",
        "Name one limiting condition for {topic}."
    ],
    "object": [
        "Name one primary function of {topic}.",
        "Name one component part of {topic}.",
        "Name one context where {topic} is commonly found.",
        "Name one material commonly used to make {topic}.",
        "Name one variation or subtype of {topic}.",
        "Name one manufacturer or origin of {topic}."
    ],
    "concept": [
        "Name one foundational principle of {topic}.",
        "Name one field where {topic} is applied.",
        "Name one historical origin of {topic}.",
        "Name one practical application of {topic}.",
        "Name one key term associated with {topic}.",
        "Name one major contributor to {topic}."
    ]
}



def generate_prompt_dataset():
    prompts = []
    prompt_id = 0
    for category, subcats in WIKI_CATEGORIES.items():
        for subcat, topics in subcats.items():
            for topic_obj in topics:
                topic = topic_obj["name"]
                topic_type = topic_obj["type"]
                templates = PROMPT_TEMPLATES.get(topic_type, [])
                advanced_templates = ADVANCED_TEMPLATES.get(topic_type, [])
                advanced_templates = advanced_templates + templates  # Include basic templates too
                for template in advanced_templates:
                    prompts.append({
                        "id": prompt_id,
                        "category": category,
                        "subcategory": subcat,
                        "topic": topic,
                        "topic_type": topic_type,
                        "template": template,
                        "prompt": template.format(topic=topic)
                    })
                    prompt_id += 1
    random.shuffle(prompts)
    return prompts

def generate_fact_check_dataset():
    prompts = []
    prompt_id = 0
    for category, subcats in WIKI_CATEGORIES.items():
        for subcat, topics in subcats.items():
            for topic_obj in topics:
                topic = topic_obj["name"]
                topic_type = topic_obj["type"]
                templates = FACT_CHECK_TEMPLATES.get(topic_type, [])
                for template in templates:
                    prompts.append({
                        "id": prompt_id,
                        "category": category,
                        "subcategory": subcat,
                        "topic": topic,
                        "topic_type": topic_type,
                        "template": template,
                        "prompt": template.format(topic=topic),
                        "style": "fact_check"
                    })
                    prompt_id += 1
    random.shuffle(prompts)
    return prompts


if __name__ == "__main__":

    prompts = generate_prompt_dataset()
    with open(os.path.join(save_folder, "evaluation_prompts_wikipedia_topics.json"), "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Generated {len(prompts)} Wikipedia-aligned evaluation prompts.")

    fact_prompts = generate_fact_check_dataset()
    with open(os.path.join(save_folder, "fact_check_prompts_wikipedia_topics.json"), "w") as f:
        json.dump(fact_prompts, f, indent=2)
    print(f"Generated {len(fact_prompts)} fact-check prompts.")
