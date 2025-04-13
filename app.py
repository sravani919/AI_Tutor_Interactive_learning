#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ✅ Import Libraries
import fitz  # PyMuPDF for handling PDFs
import torch
import json
import pandas as pd
import random
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests  
import matplotlib.pyplot as plt
import numpy as np
import re
from IPython.display import display, HTML
import html
import threading
import time
from ipywidgets import RadioButtons, HBox, VBox, Label


# ✅ Load AI Model (Mistral-7B)
# model_name = "mistralai/Mistral-7B-v0.1"

# Change this
model_name = "gpt2"  # instead of "mistralai/Mistral-7B-v0.1"

# And skip quantization for simple CPU deployment
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=quantization_config,
#     device_map="auto",
#     torch_dtype=torch.float16,
# )



print("✅ AI Model Loaded!")

# ✅ Create AI Chat Pipeline
mistral_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.9,
     top_p=0.95,
    do_sample=True,
)




# ✅ Wrap Pipeline in LangChain LLM
llm = HuggingFacePipeline(pipeline=mistral_pipeline)
print("✅ AI Integrated with LangChain!")

# ✅ Load Chapter Dataset
df_grouped = pd.read_csv("Merged_Chapter_Dataset.csv")

chapter_summaries = {}
chapter_questions = {}
chapter_answers = {} 

def clean_answer_from_question(question, answer):
    import difflib

    q_words = question.lower().split()
    a_words = answer.strip().split()

    # Strip punctuation from both question and answer words
    q_set = set(w.strip(".,?") for w in q_words)

    # Identify where the answer starts to differ significantly
    start_index = 0
    for i, word in enumerate(a_words):
        clean_word = word.lower().strip(".,?")
        if clean_word not in q_set:
            break
        start_index += 1

    # Remove repeated portion
    trimmed = a_words[start_index:]
    cleaned = " ".join(trimmed).strip()

    # Fallback if answer becomes empty or too short
    if not cleaned or len(cleaned.split()) <= 3:
        cleaned = "It refers to " + " ".join(a_words)

    # Capitalize first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned.rstrip(". ")



for _, row in df_grouped.iterrows():
    chapter = str(row["chapter"])
    chapter_content = str(row["Chapter Content"])
    questions = eval(row["Questions"]) if isinstance(row["Questions"], str) else row["Questions"]
    answers = eval(row["Answers"]) if isinstance(row["Answers"], str) else row["Answers"]

    chapter_summaries[chapter] = chapter_content if chapter_content else "No summary available."
    chapter_questions[chapter] = questions[:5] if questions else []

    cleaned = [
        clean_answer_from_question(q, a)
        for q, a in zip(questions[:5], answers[:5])
    ] if questions and answers else []

    chapter_answers[chapter] = cleaned

    # # ✅ Print a few samples for verification
    # print(f"\n📘 Chapter: {chapter}")
    # for q, a in zip(chapter_questions[chapter], chapter_answers[chapter]):
    #     print(f"Q: {q}")
    #     print(f"A: {a}")
    #     print("---")



# ✅ Create FAISS Vector Store for Retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_store = FAISS.from_texts(list(chapter_questions.keys()), embeddings)
faiss_store.add_texts([" ".join(a) if isinstance(a, list) else str(a) for a in chapter_questions.values()])
print("✅ FAISS Vector Store Created!")

# ✅ AI Chat Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=faiss_store.as_retriever(),
    memory=memory
)
print("✅ AI Tutor Ready!")

# ✅ Google API Setup for Image & Additional Info
GOOGLE_API_KEY = "AIzaSyA-_VhTAlMXZPdKI__qN4rydZflIh7oAP4"
GOOGLE_CSE_ID = "c00f584f8b0c54842"

def fetch_more_info(topic):
    """Fetches relevant business cases, industry news, and research papers using Google Search API."""
    try:
        search_url = f"https://www.googleapis.com/customsearch/v1?q={topic.replace(' ', '+')}+business+case+study+industry+news+latest+trends&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(search_url)
        data = response.json()

        links = []
        if "items" in data and len(data["items"]) > 0:
            for item in data["items"][:5]:  # Get top 5 relevant articles
                title = item.get("title", "Read More")
                link = item.get("link", "#")
                links.append(f"<li><a href='{link}' target='_blank'>{title}</a></li>")

        if links:
            return f"""
                <h3 style='color:blue;'>🌐 Additional Learning Resources</h3>
                <p>Here are some external resources to explore more about <b>{topic}</b>:</p>
                <ul>{''.join(links)}</ul>
            """
        else:
            return "<p>⚠️ No additional resources found. Let's focus on AI-generated insights!</p>"

    except Exception as e:
        print(f"⚠️ Error fetching additional information: {e}")
        return "<p>⚠️ Error fetching additional resources. Try again later.</p>"



def fetch_relevant_image(topic):
    """Fetches a relevant image using Google Custom Search API."""
    try:
        refined_query = f"{topic} infographic OR concept OR business example OR diagram"
        search_url = f"https://www.googleapis.com/customsearch/v1?q={refined_query.replace(' ', '+')}&searchType=image&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(search_url)
        data = response.json()

        if "items" in data and len(data["items"]) > 0:
            img_url = data["items"][0]["link"]
            image_response = requests.get(img_url)
            img = Image.open(BytesIO(image_response.content))
            return img  

    except UnidentifiedImageError:
        return None  



def clean_chapter_name(chapter_name):
    """Removes leading numbers and trims spaces from chapter names."""
    return re.sub(r'^\d+(\.\d+)?\s*', '', chapter_name).strip()










def get_industry(chapter_name):
    """Returns the corresponding industry for a given chapter."""

    industry_mapping = {
        "Technology & Computing": [
            "Computing from Inception to Today", "Computer Hardware and Networks",
            "The Internet, Cloud Computing, and the Internet of Things"
        ],
        "Cybersecurity & Ethics": [
            "Safety, Security, Privacy, and the Ethical Use of Technology"
        ],
        "Data Analysis & BI": [
            "Data Tables and Ranges", "Data Analysis Charts", "PivotTables", "What-If Analysis",
            "Statistical Functions"
        ],
        "Accounting & Finance": [
            "Basic Accounting", "Financial Functions in Microsoft Excel",
            "Auditing Formulas and Fixing Errors", "Integrating Microsoft Excel and Accounting Programs"
        ],
        "Databases & Data Management": [
            "What Is a Database?", "Querying a Database", "Maintaining Records in a Database",
            "Creating Reports in Microsoft Access", "Advanced Queries in Microsoft Access"
        ],
        "Business Collaboration": [
            "Microsoft 365: Collaboration and Integration", "Essentials of Google Workspace",
            "Communication and Calendar Applications", "Collaboration"
        ],
        "Document Processing": [
            "Microsoft Word: Integration with Microsoft Excel and Microsoft Access",
            "Formatting Document Layout in Microsoft Word", "Collaborative Editing and Reviewing in Microsoft Word",
            "Document Design", "Navigating Google Docs", "Collaborative Editing and Reviewing in Google Docs"
        ],
        "Presentation & Design": [
            "Presentation and Design Essentials", "Designing a Presentation in Microsoft PowerPoint",
            "Formatting Microsoft PowerPoint Slides: Layout and Design Principles",
            "Adding Visuals and Features to Microsoft PowerPoint Slides"
        ],
        "Content & Digital Marketing": [
            "Search Engine Optimization", "Social Media in Business",
            "What Are Content Management Systems?", "Common Content Management Systems",
            "Creating Content with a Content Management System"
        ],
        "Spreadsheet & Automation": [
            "Microsoft Excel Basics", "Google Sheets Basics", "Calculations and Basic Formulas in Microsoft Excel",
            "Formatting and Templates in Microsoft Excel", "Using Arithmetic, Statistical, and Logical Functions"
        ]
    }

    # Reverse map chapters to their industry
    chapter_to_industry = {chapter: industry for industry, chapters in industry_mapping.items() for chapter in chapters}

    return chapter_to_industry.get(chapter_name, "General Business")

import random

def generate_business_case_logic_v9(chapter):
    """Generates a structured business case dynamically with industry-specific logic and real-world statistics."""

    # ✅ Clean chapter name
    cleaned_chapter = clean_chapter_name(chapter)

    # ✅ Get the industry
    industry = get_industry(cleaned_chapter)

    # ✅ Define alternative phrasing for natural-sounding text
    alternative_wordings = {
        "Customizing Reports": "tailored business reporting",
        "Financial Functions in Microsoft Excel": "advanced financial modeling",
        "Integrating Microsoft Excel and Accounting Programs": "financial automation tools",
        "Data Analysis Charts": "data visualization techniques",
        "What Is a Database?": "database management systems",
        "Collaboration": "collaborative business platforms"
    }

    # Use alternative phrasing if available, otherwise use the cleaned chapter name
    alt_wording = alternative_wordings.get(cleaned_chapter, cleaned_chapter.lower())

    # ✅ Generate a company name
    company_names = ["AlphaCorp", "Beta Enterprises", "Gamma Solutions", "Delta Holdings", "Epsilon Ventures"]
    company = random.choice(company_names)

    # ✅ Realistic Industry Trends & Statistics
    industry_trends = {
        "General Business": {
            "fact": "Studies show that 60% of companies struggle with inaccurate business reports, leading to poor decision-making.",
            "financial_impact": "Companies that invest in real-time reporting solutions see a 30% improvement in operational efficiency."
        },
        "Accounting & Finance": {
            "fact": "79% of financial professionals believe that automation reduces human errors in financial reporting.",
            "financial_impact": "Organizations that integrate financial automation tools see a 25% reduction in accounting errors."
        },
        "Data Analysis & BI": {
            "fact": "Businesses that implement data visualization techniques make decisions 5x faster than those using traditional reports.",
            "financial_impact": "A well-optimized business intelligence strategy can increase revenue by up to 15%."
        }
    }

    # ✅ Get industry-specific statistics or use a default fallback
    trend_data = industry_trends.get(industry, {
        "fact": "Companies that optimize workflow automation reduce inefficiencies by 40%.",
        "financial_impact": "Process automation solutions can save businesses up to $2.3 million annually."
    })

    # ✅ Generate Business Case
    business_case = {
        "Executive Summary": f"""
            {company}, a leading firm in {industry}, faced significant challenges in {cleaned_chapter}. 
            {trend_data["fact"]} However, outdated processes caused inefficiencies, leading to errors and slow decision-making.
            This business case explores how {company} leveraged {alt_wording} solutions to optimize its operations.
        """,

        "Problem Statement": f"""
            {company} relied on outdated reporting methods that led to inconsistencies, delayed insights, 
            and poor data accuracy. As a result, key stakeholders lacked the real-time data needed for critical business decisions.
        """,

        "Why is This Important?": f"""
            If left unresolved, these inefficiencies would cause significant revenue losses and operational bottlenecks.
            {trend_data["financial_impact"]} To remain competitive, organizations must implement streamlined reporting solutions.
        """,

        "Proposed Solution": f"""
            To address these issues, {company} adopted modern {alt_wording} tools, integrating AI-driven automation to 
            improve accuracy and reduce report generation time by 50%. By leveraging real-time data visualization 
            and predictive analytics, the company enhanced its decision-making capabilities.
        """,

        "Business Objectives": [
            f"Improve operational efficiency in {alt_wording}-related processes by 80%.",
            f"Reduce manual errors in reporting by 65% through automation.",
            f"Ensure compliance with industry standards and data security regulations.",
            f"Enhance real-time decision-making by reducing report generation time by 50%."
        ],

        "Scope & Impact": f"""
            The project involved collaboration between the IT department, finance team, and external consultants. 
            Key challenges included data migration and employee training, which were managed through a structured 
            implementation plan and phased rollout.
        """,

        "Financials & Timeline": f"""
            The company invested $200,000 in {alt_wording} upgrades, with a projected return of investment (ROI) of 3.5x 
            over three years. Estimated annual cost savings exceeded $500,000 due to efficiency improvements.
        """,

        "Conclusion & Next Steps": f"""
            With the successful adoption of {alt_wording}, {company} significantly improved its operational efficiency. 
            Future steps include further integration of AI-powered analytics to enhance forecasting accuracy.
        """
    }

    return business_case

def display_business_case(chapter):
    """Formats and displays the business case in a structured format."""

    # Generate the business case
    business_case = generate_business_case_logic_v9(chapter)


    formatted_case = f"""
        <h3 style='color:blue;'>📊 Business Scenario:</h3>
        <p>{business_case["Executive Summary"]}</p>

        <h3 style='color:blue;'>📝 Task:</h3>
        <p>Analyze how <b>{chapter}</b> was used to improve business decision-making.</p>

        <h3 style='color:blue;'>📌 Problem Statement:</h3>
        <p>{business_case["Problem Statement"]}</p>

        <h3 style='color:blue;'>⚠️ Why is This Important?</h3>
        <p>{business_case["Why is This Important?"]}</p>

        <h3 style='color:blue;'>💡 Proposed Solution:</h3>
        <p>{business_case["Proposed Solution"]}</p>

        <h3 style='color:blue;'>🎯 Business Objectives:</h3>
        <ul>
            <li>{business_case["Business Objectives"][0]}</li>
            <li>{business_case["Business Objectives"][1]}</li>
            <li>{business_case["Business Objectives"][2]}</li>
            <li>{business_case["Business Objectives"][3]}</li>
        </ul>

        <h3 style='color:blue;'>💰 Financials & Timeline:</h3>
        <p>{business_case["Financials & Timeline"]}</p>

        <h3 style='color:blue;'>🚀 Conclusion & Next Steps:</h3>
        <p>{business_case["Conclusion & Next Steps"]}</p>
    """

    return formatted_case




# ✅ Fetch Business Story from Google or Generate AI Story

def fetch_google_case_studies(topic):
    """Fetches multiple relevant business case studies from Google Search API."""
    try:
        search_query = f"{topic} real-world business case OR industry application OR data-driven decision making"
        search_url = f"https://www.googleapis.com/customsearch/v1?q={search_query.replace(' ', '+')}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(search_url)
        data = response.json()

        if "items" in data and len(data["items"]) > 0:
            return [item["link"] for item in data["items"][:3]]  # Get the top 3 case study links

        return None  # No relevant cases found

    except Exception as e:
        print(f"⚠️ Google Business Case Fetch Error: {e}")
        return None




# ✅ Fetch Business Story from Google or Generate AI Story


def generate_dynamic_storytelling(chapter):
    """Generates a compelling and dynamic storytelling experience related to any chapter."""

    # ✅ Clean the chapter name
    cleaned_chapter = clean_chapter_name(chapter)

    # ✅ Define a Random Protagonist (AI-Generated)
    protagonist_options = [
        {"name": "Mia", "role": "a supply chain manager"},
        {"name": "Alex", "role": "a financial analyst"},
        {"name": "Jordan", "role": "a software engineer"},
        {"name": "Sophia", "role": "an IT director"},
        {"name": "Leo", "role": "a business consultant"},
        {"name": "Lisa", "role": "a marketing strategist"},
        {"name": "Chris", "role": "a data scientist"},
    ]
    protagonist = random.choice(protagonist_options)

    # ✅ Dynamic Challenge Related to the Chapter
    chapter_challenges = [
        f"is struggling with {cleaned_chapter.lower()} concepts, making their daily tasks inefficient.",
        f"is facing difficulties using {cleaned_chapter.lower()} in real-world business scenarios.",
        f"is overwhelmed by complex {cleaned_chapter.lower()} calculations, slowing down productivity.",
        f"finds it challenging to apply {cleaned_chapter.lower()} for data-driven decision-making.",
        f"is spending hours manually handling {cleaned_chapter.lower()}, leading to errors and inefficiencies.",
        f"is looking for a better way to leverage {cleaned_chapter.lower()} for business impact."
    ]
    challenge = random.choice(chapter_challenges)

    # ✅ Introduce a Guide (AI-Generated Expert)
    expert_guides = [
        "Michael, a data analytics expert with 10+ years of experience.",
        "David, a senior software developer specializing in automation.",
        "Emily, a business intelligence consultant who optimizes workflows.",
        "Lisa, a financial expert known for simplifying complex data models.",
        "Chris, an AI strategist helping businesses adopt modern solutions."
    ]
    guide = random.choice(expert_guides)

    # ✅ Define the Transformation (Solution Related to the Chapter)
    transformations = [
        f"learns how to apply {cleaned_chapter.lower()} to optimize their workflow.",
        f"automates tasks using {cleaned_chapter.lower()}, reducing manual effort by 50%.",
        f"adopts new techniques in {cleaned_chapter.lower()}, improving efficiency and accuracy.",
        f"implements AI-driven solutions for {cleaned_chapter.lower()}, leading to real-time insights.",
        f"streamlines processes with {cleaned_chapter.lower()}, cutting report generation time in half."
    ]
    transformation = random.choice(transformations)

    # ✅ Define the Resolution (Outcome Metrics)
    productivity_boost = random.randint(20, 50)
    revenue_boost = random.randint(5, 15)
    resolution = f"As a result, {protagonist['name']} saw a dramatic improvement in efficiency, leading to a {productivity_boost}% boost in productivity and a {revenue_boost}% increase in revenue."

    # ✅ Compile the Storytelling Output
    story = f"""
    <h3 style='color:darkblue;'>📖 AI-Powered Business Storytelling</h3>

    <h4 style='color:black;'>🔹 Act 1: Introducing the Protagonist & Their Challenge</h4>
    <p><b>Protagonist:</b> {protagonist['name']}, {protagonist['role']} who {challenge}</p>
    <p style="color:gray;">"{protagonist['name']} felt frustrated and overwhelmed, spending long hours trying to find a solution with no clear insights."</p>
    <p><i>“There must be a better way,”</i> {protagonist['name']} thought.</p>

    <h4 style='color:black;'>🔹 Act 2: Enter the Guide & The Transformation</h4>
    <p><b>Guide:</b> Seeking solutions, {protagonist['name']} consulted {guide}, who had experience solving similar challenges.</p>
    <p><b>Solution:</b> With expert guidance, they <b>{transformation}</b></p>

    <h4 style='color:black;'>🔹 Act 3: Resolution & Success</h4>
    <p><b>Outcome:</b> {resolution}</p>
    <p style="color:gray;">"This solution transformed the way {protagonist['name']} worked, enabling faster decision-making and optimized efficiency!"</p>

    <h3 style='color:green;'>🔹 Key Takeaways</h3>
    <ul>
        <li>📊 Data-driven solutions significantly improve decision-making.</li>
        <li>🚀 Automating processes can lead to time savings and efficiency gains.</li>
        <li>💡 Companies that embrace innovation stay competitive and future-proof their business.</li>
    </ul>

    <h3 style='color:darkblue;'>🤔 Reflection Questions</h3>
    <ul>
        <li>How does this situation relate to your own work?</li>
        <li>What steps would you take in a similar challenge?</li>
        <li>Could automation improve your workflow?</li>
    </ul>

    <h3 style='color:blue;'>🚀 Ready to optimize your workflow?</h3>
    <p>"Imagine your team making faster, data-driven decisions like {protagonist['name']}! What challenges do you face today? Let’s explore solutions together!"</p>

    <h3 style='color:darkblue;'>📚 Would you like to explore another chapter?</h3>
    """

    return story


def load_and_display_challenges(industry):
    industry_specific_challenges = {
        "Data Analysis & BI": [
            "Use a dataset to generate a report based on statistical analysis and identify trends.",
            "Explore how business intelligence dashboards improve real-time decision-making."
        ],
        "Accounting & Finance": [
            "Build a financial model using Excel to simulate different economic scenarios.",
            "Analyze a set of financial statements to determine the health of a business."
        ],
        "Technology & Computing": [
            "Develop a simple software application that uses basic data structures.",
            "Create a database schema that represents a small business inventory system."
        ],
        "General Business": [
            "Prepare a marketing plan for a new product launch.",
            "Conduct a SWOT analysis for a chosen company."
        ]
    }
    challenges = industry_specific_challenges.get(industry, ["General challenge related to business strategies."])

    # Display the challenges
    print("\nChallenges for", industry, "Industry:")
    for challenge in challenges:
        print("- ", challenge)


#Challenges        
#Flipcard    


# Flashcard State
flashcards = []
current_index = 0
flipped = False

# UI Elements
flashcard_label = widgets.Label()
flip_button = widgets.Button(description="🔄 Flip Card", button_style="info")
next_card_btn = widgets.Button(description="➡️ Next", button_style="success")
restart_btn = widgets.Button(description="🔁 Restart", button_style="warning")
progress_label = widgets.Label()
correct_btn = widgets.Button(description="👍 I got it!", button_style="success")
incorrect_btn = widgets.Button(description="👎 I missed it", button_style="danger")


def show_flashcard(_=None):
    global current_index, flipped
    flipped = False
    with output:
        clear_output(wait=True)

        if current_index < len(flashcards):
            question, _ = flashcards[current_index]
            flashcard_label.value = f"🃏 {question} (Tap to Flip)"
            progress_label.value = f"Progress: {current_index + 1} / {len(flashcards)}"

            display(HTML("<h3>🧠 Flashcard Challenge</h3>"))
            display(flashcard_label)
            display(widgets.HBox([flip_button, next_card_btn]))
            display(progress_label)
            display(widgets.HBox([correct_btn, incorrect_btn]))
            display(restart_btn)

            # ✅ Additional Learning Resources while playing
            selected_chapter = chapter_dropdown.value
            if selected_chapter != "Select a Chapter":
                more_info_html = fetch_more_info(selected_chapter)
                display(HTML(more_info_html))

            # ✅ Next actions (display during flashcards, not just at the end)
            display(HTML("<h4 style='margin-top:20px;'>✅ What would you like to do next?</h4>"))
            display(next_action_radio, next_button)

        else:
            flashcard_label.value = "🎉 You've completed all flashcards for this chapter!"
            display(flashcard_label)
            display(HTML("<hr><h4>✅ What would you like to do next?</h4>"))

            update_next_action_options("Flashcards")
            display(next_action_radio, next_button)



def flip_flashcard(_):
    global flipped
    with output:
        if not flipped and current_index < len(flashcards):
            _, answer = flashcards[current_index]
            flashcard_label.value = f"✅ {answer}"
            flipped = True
            display(HTML("<p style='color:green;'>+5 XP! Keep going 💪</p>"))
        elif flipped:
            show_flashcard()

def next_flashcard(_):
    global current_index
    if current_index < len(flashcards):
        current_index += 1
    show_flashcard()

def restart_flashcards(_):
    global current_index
    current_index = 0
    random.shuffle(flashcards)
    show_flashcard()

def mark_correct(_):
    award_xp("🧠 Flashcards (Flip)")
    next_flashcard(None)


def mark_incorrect(_):
    next_flashcard(None)



##designing MCQs


# Global state for MCQs
# MCQ Quiz State
# Globals for MCQ tracking
mcq_score = 0
mcq_questions_asked = 0
mcq_total_questions = 5
mcq_correct_answer = ""
mcq_buttons = []



mcq_feedback_label = widgets.Label()
feedback = widgets.HTML()
mcq_radio = None  # Declare globally for reuse



def generate_generic_mcq_options(question, correct_answer, chapter_summary=""):
    """
    Generates 4 distinct MCQ options: correct + 3 diverse, non-overlapping distractors.
    """
    base = re.split(r"[.,]", correct_answer)[0].strip()
    words = base.split()
    main_idea = " ".join(words[:5]) if len(words) > 2 else base

    distractor_templates = [
        f"{main_idea} is somewhat related but lacks the key details.",
        f"{main_idea} is often confused with formatting options like fonts or colors.",
        f"{main_idea} refers to a different feature not related to this context.",
        f"{main_idea} is part of data entry, not analysis.",
        f"{main_idea} is about aesthetics, not insights.",
        f"{main_idea} helps organize data but doesn't summarize it.",
        f"{main_idea} works with raw data, not summary metrics.",
    ]

    # Remove any that accidentally match the correct answer
    unique_distractors = list(set([
        d for d in distractor_templates if d.strip().lower() != correct_answer.strip().lower()
    ]))

    # Randomly select 3 unique distractors
    random.shuffle(unique_distractors)
    selected_distractors = unique_distractors[:3]

    # Combine and shuffle
    all_options = [correct_answer.strip()] + selected_distractors
    random.shuffle(all_options)

    return all_options





def clean_answer_text(text):
    return re.split(r"[,.]", text)[0].strip().capitalize()


def show_mcq_question(_=None):
    global mcq_questions_asked, mcq_score, mcq_radio

    with output:
        clear_output(wait=True)

        selected_chapter = chapter_dropdown.value
        if selected_chapter not in chapter_questions:
            display(HTML("<p style='color:red;'>⚠️ No questions found for this chapter.</p>"))
            return

        q_list = chapter_questions[selected_chapter]
        a_list = chapter_answers[selected_chapter]

        if mcq_questions_asked >= mcq_total_questions or not q_list:
            display(HTML(f"<h3>🎉 Quiz Complete!</h3><p>Final Score: <b>{mcq_score}/{mcq_total_questions * 10}</b></p>"))
            update_next_action_options("MCQs")
            display(next_action_radio, next_button)
            return

        idx = mcq_questions_asked % len(q_list)
        question_text = f"{q_list[idx]}"
        correct = a_list[idx].strip()

        options = generate_generic_mcq_options(q_list[idx], correct)
        question_text_clean = html.unescape(question_text)

        feedback_box = widgets.HTML()
        mcq_next_button = widgets.Button(description="➡️ Next", button_style="success", disabled=True)

        def on_next_mcq(_):
            global mcq_questions_asked
            mcq_questions_asked += 1
            show_mcq_question()

        mcq_next_button.on_click(on_next_mcq)

        #from IPython.display import HTML

#         display(HTML("""
#         <style>
#         /* 🔧 Fix MCQ Layout for long option text */
#         .widget-radio-box label {
#             display: flex !important;
#             align-items: flex-start !important;
#             flex-direction: row !important;
#             gap: 12px;
#             padding: 8px 10px;
#             white-space: normal !important;
#             word-break: break-word;
#             max-width: 100%;
#             line-height: 1.6em;
#             font-size: 15px;
#         }

#         /* Radio circle stays top-aligned */
#         .widget-radio-box input[type="radio"] {
#             margin-top: 6px !important;
#             flex-shrink: 0;
#         }

#         /* Ensure text wraps cleanly */
#         .widget-radio-box span {
#             white-space: normal !important;
#             word-break: break-word;
#             flex-grow: 1;
#         }
#         </style>
#         """))


        # ✅ Create the radio buttons
        mcq_radio = widgets.RadioButtons(
            options=[f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)],
            layout=widgets.Layout(width='100%'),
            style={'description_width': '0px'},
            value=None
        )

        submit_button = widgets.Button(description="Submit", button_style="primary")

        def on_submit_click(_):
            selected = mcq_radio.value
            if not selected:
                feedback_box.value = "<p style='color:red;'>❌ Please select an option first.</p>"
                return

            selected_text = selected.split('. ', 1)[1]
            if selected_text.strip().lower() == correct.strip().lower():
                feedback_box.value = "<div style='color:green; font-weight:bold;'>✅ Correct! +10 XP</div>"
                award_xp("✅ MCQ Quiz")
                global mcq_score
                mcq_score += 10
            else:
                feedback_box.value = f"""
                    <div style='color:red; font-weight:bold;'>❌ Incorrect.</div>
                    <div><span style='color:green;'>✔ Correct Answer:</span> {correct}</div>
                """
            mcq_next_button.disabled = False

        submit_button.on_click(on_submit_click)

        # ✅ Display all MCQ UI
        display(HTML("<h3 style='color:#1E90FF;'>📘 Multiple Choice Question</h3>"))
        display(HTML(f"<div style='font-weight: bold; font-size: 22px;'>{question_text_clean}</div>"))
        display(mcq_radio)
        display(submit_button)
        display(feedback_box)
        display(mcq_next_button)
        display(HTML(f"<p><b>Progress:</b> {mcq_questions_asked + 1}/{mcq_total_questions}</p>"))

        # 🔎 More learning resources
        more_info_html = fetch_more_info(selected_chapter)
        display(HTML(more_info_html))

        # 🔁 Navigation options
        display(HTML("<hr><h4>✅ What would you like to do next?</h4>"))
        update_next_action_options("MCQs")
        display(next_action_radio, next_button)





#fill_in_the_blanks        
# Global state
# === Fill in the Blank Multi-Screen Typing Game === #

# === Interactive Fill-in-the-Blanks Game ===

interactive_blocks = []
interactive_index = 0
interactive_score = 0
interactive_lives = 3
interactive_total = 0

def parse_text_with_blanks(raw_text):
    """Extracts blanks wrapped in [ ] and returns processed text with placeholders."""
    blanks = re.findall(r"\[(.+?)\]", raw_text)
    processed_text = re.sub(r"\[(.+?)\]", "____", raw_text)
    return processed_text, blanks

def generate_interactive_blocks():
    selected_chapter = chapter_dropdown.value
    answers = chapter_answers.get(selected_chapter, [])
    all_answers = [a for sublist in chapter_answers.values() for a in sublist if isinstance(sublist, list)]

    blocks = []

    for a in answers:
        sentence = a.strip()
        words = re.findall(r'\b\w+\b', sentence)

        # Pick a word over 4 characters as the blank
        # ✅ Filter out generic words like 'purpose', 'benefits', etc.
        generic_words = {
            "purpose", "benefits", "advantages", "method", "steps", "formatting", 
            "data", "information", "process", "important", "feature", "common", 
            "value", "reason", "result", "approach"
        }

        # ✅ Choose the first important word that's NOT a generic word
        keyword = next(
            (w for w in words if len(w) > 4 and w.lower() not in generic_words),
            None
        )

        if not keyword:
            continue

        blanked_sentence = sentence.replace(keyword, "____", 1)

        # Distractors: other words from other answers
        distractor_pool = list(set(
            w for ans in all_answers for w in re.findall(r'\b\w+\b', ans)
            if w.lower() != keyword.lower() and len(w) > 4
        ))
        distractors = random.sample(distractor_pool, min(3, len(distractor_pool)))

        word_bank = list(set([keyword] + distractors))
        random.shuffle(word_bank)

        blocks.append({
            "text": blanked_sentence,
            "answers": [keyword],
            "word_bank": word_bank
        })

    print(f"✅ Generated {len(blocks)} answer-based fill-in-the-blank blocks.")
    return blocks[:5]


def start_interactive_fill_blank_game():
    global interactive_blocks, interactive_index, interactive_score, interactive_lives, interactive_total
    interactive_blocks = generate_interactive_blocks()

    if not interactive_blocks:
        with output:
            clear_output()
            display(HTML("<h3 style='color:red;'>⚠️ Not enough data to generate a Fill-in-the-Blank challenge for this chapter.</h3>"))
            update_next_action_options("Fill-in-the-Blank")
            display(next_action_radio, next_button)
        return  # ❌ Do not continue if empty

    interactive_index = 0
    interactive_score = 0
    interactive_lives = 3
    interactive_total = len(interactive_blocks)
    render_interactive_block()
print(f"🔍 Generated {len(interactive_blocks)} interactive blocks.")


def render_interactive_block():
    global interactive_index, interactive_score, interactive_lives

    block = interactive_blocks[interactive_index]

    with output:
        clear_output()

        display(HTML(f"<h3>✏️ Interactive Fill in the Blanks (Block {interactive_index + 1}/{len(interactive_blocks)})</h3>"))


        display(HTML(f"<p><b>Sentence:</b> {block['text']}</p>"))
        display(HTML(f"<p><b>Lives:</b> {'❤️' * interactive_lives} &nbsp;&nbsp; <b>Score:</b> {interactive_score}</p>"))

        dropdown = widgets.Dropdown(
            options=["Select"] + block["word_bank"],
            description="Blank 1:"
        )
        submit_btn = widgets.Button(description="Submit", button_style="primary")
        feedback_label = widgets.Label()


        def check_answer(_):
            user_input = dropdown.value.strip().lower()
            correct = block["answers"][0].strip().lower()

            global interactive_lives, interactive_score


            if user_input == correct:
                feedback_label.value = "✅ Correct! +10 XP"
                interactive_score += 10
                award_xp("✏️ Fill in the Blank")
            else:
                interactive_lives -= 1
                feedback_label.value = f"❌ Incorrect. Lives left: {'❤️' * interactive_lives}"

            if interactive_lives == 0:
                display(HTML("<h3 style='color:red;'>💀 Game Over!</h3>"))
                display_final_fill_blank_result()
                return

            next_btn.disabled = False

        submit_btn.on_click(check_answer)

        next_btn = widgets.Button(description="➡️ Next", button_style="success", disabled=True)

        def load_next(_):
            global interactive_index
            interactive_index += 1
            if interactive_index < len(interactive_blocks):
                render_interactive_block()
            else:
                display_final_fill_blank_result()

        next_btn.on_click(load_next)

        display(dropdown, widgets.HBox([submit_btn, next_btn]), feedback_label)


                # ✅ Show global navigation (same as other modes)
        display(HTML("<hr><h4>✅ What would you like to do next?</h4>"))
        update_next_action_options("Fill-in-the-Blank")
        display(next_action_radio, next_button)



        def display_final_fill_blank_result():
            selected_chapter = chapter_dropdown.value
            summary = chapter_summaries.get(selected_chapter, "No summary available.")

            with output:
                clear_output()

                # ✅ Show Additional Info
                more_info_html = fetch_more_info(selected_chapter)
                display(HTML(more_info_html))

                # ✅ Game Score
                display(HTML(f"<h3>🎉 Game Complete!</h3><p><b>Final Score:</b> {interactive_score}</p>"))

                # ✅ Show all sentences with correct answers
                display(HTML("<h4>📘 Review Your Questions & Answers:</h4>"))
                for block in interactive_blocks:
                    sentence = block["text"]
                    correct_word = block["answers"][0]
                    reviewed_sentence = sentence.replace("____", f"<b style='color:green;'>{correct_word}</b>")
                    display(HTML(f"🔹 {reviewed_sentence}"))

                # ✅ Show chapter summary for reinforcement
                display(HTML(f"<h4>📖 Chapter Summary:</h4><p>{summary}</p>"))

                # ✅ Offer next steps
                update_next_action_options("Fill-in-the-Blank")
                display(next_action_radio, next_button)



#Match the answer logic 
def show_matching_game():
    selected_chapter = chapter_dropdown.value
    questions = chapter_questions.get(selected_chapter, [])
    answers = chapter_answers.get(selected_chapter, [])

    if not questions or not answers or len(questions) != len(answers):
        with output:
            clear_output()
            display(HTML("<p style='color:red;'>⚠️ Not enough matching pairs found for this chapter.</p>"))
            update_next_action_options("Match the Answers")
            display(next_action_radio, next_button)
        return

    with output:
        clear_output()
        display(HTML("<h3>🧩 Match the Answers (Dropdown Version)</h3>"))

        # Shuffle the answer options
        shuffled_answers = answers.copy()
        random.shuffle(shuffled_answers)

        dropdowns = []
        form_items = []

        for i, question in enumerate(questions):
            dropdown = widgets.Dropdown(
                options=["Select an Answer"] + shuffled_answers,
                description=f"A{i+1}:",
                layout=widgets.Layout(width="100%")
            )
            q_html = widgets.HTML(value=f"<b>Q{i+1}:</b> {question}")
            form_items.append((q_html, dropdown))

            dropdowns.append((dropdown, answers[i]))

        check_btn = widgets.Button(description="✅ Check Matches", button_style="success")
        result_box = widgets.Output()
        show_correct_btn = widgets.Button(description="👁️ Show Correct Answers", button_style="warning")
        show_correct_btn.layout.display = 'none'  # Hidden by default

        def check_answers(_):
            correct = 0
            incorrect = []
            result_box.clear_output()

            with result_box:
                html_output = """
                <style>
                .match-summary-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                    font-family: Arial, sans-serif;
                }
                .match-summary-table th, .match-summary-table td {
                    border: 1px solid #ddd;
                    padding: 14px;
                    text-align: left;
                    vertical-align: top;
                }
                .match-summary-table th {
                    background-color: #f0f8ff;
                    color: #333;
                }
                .correct {
                    background-color: #e6ffed;
                }
                .incorrect {
                    background-color: #fff0f0;
                }
                .your-answer {
                    font-weight: bold;
                    color: #003366;
                }
                .correct-answer {
                    font-weight: bold;
                    color: #006400;
                }
                .incorrect-answer {
                    font-weight: bold;
                    color: #b30000;
                }
                </style>

                <h4>📊 <b>Match Results Summary</b></h4>
                <table class='match-summary-table'>
                    <tr>
                        <th>Question</th>
                        <th>Your Answer</th>
                        <th>Correct Answer</th>
                        <th>Result</th>
                    </tr>
                """

                for idx, (dropdown, correct_answer) in enumerate(dropdowns):
                    selected = dropdown.value or "❓ No Answer"
                    is_correct = selected == correct_answer

                    row_class = "correct" if is_correct else "incorrect"
                    icon = "✅" if is_correct else "❌"
                    result_text = "Correct" if is_correct else "Incorrect"

                    if is_correct:
                        dropdown.layout.border = "2px solid green"
                        correct += 1
                    else:
                        dropdown.layout.border = "2px solid red"
                        incorrect.append(idx)

                    html_output += f"""
                    <tr class="{row_class}">
                        <td><b>Q{idx + 1}</b></td>
                        <td class="your-answer">{icon} {selected}</td>
                        <td class="correct-answer">{'✅' if is_correct else f'<span class="incorrect-answer">✔ {correct_answer}</span>'}</td>
                        <td><b>{icon} {result_text}</b></td>
                    </tr>
                    """

                html_output += "</table>"

                summary = f"<p style='margin-top:15px; font-size:16px;'><b>🎯 Score:</b> You matched {correct} out of {len(dropdowns)} correctly.</p>"
                xp_msg = award_xp("🧩 Match the Answers") if correct == len(dropdowns) else "💡 Partial credit! Learn from mistakes and try again!"
                xp_line = f"<p><b>🏆 {xp_msg}</b></p>"

                display(HTML(html_output + summary + xp_line))
                show_correct_btn.layout.display = 'inline-block' if incorrect else 'none'



        def show_correct(_):
            result_box.clear_output()
            with result_box:
                display(HTML("<h4>✅ Correct Answers Only</h4>"))
                for idx, (dropdown, correct_answer) in enumerate(dropdowns):
                    display(HTML(f"""
                        <div style='padding:10px; border-left:5px solid #1E90FF; margin-bottom:15px; background:#f9f9f9;'>
                            <b>Q{idx+1}:</b> {questions[idx]}<br>
                            <b style='color:green;'>Answer:</b> {correct_answer}
                        </div>
                    """))


        check_btn.on_click(check_answers)
        show_correct_btn.on_click(show_correct)

        form_items_styled = []
        for q_label, dropdown in form_items:
            container = widgets.VBox([
                q_label,
                widgets.Box([dropdown], layout=widgets.Layout(margin="0 0 20px 0"))
            ])
            form_items_styled.append(container)

            form = widgets.VBox(form_items_styled)

        display(form, check_btn, result_box, show_correct_btn)

        update_next_action_options("Match the Answers")
        display(next_action_radio, next_button)


##Timedlogic

# Global for countdown management
timer_thread = None
timer_running = False
timer_seconds = 15
timer_label = widgets.Label()
timer_stop_event = threading.Event()
# Reset for next round
timed_question_count = 0
timed_question_score = 0
timed_question_total = 5  # total questions in the timed round




def show_timed_question():
    global timer_running, timer_thread, timer_label, timer_stop_event
    global timed_question_count, timed_question_score, timed_question_total

    with output:
        clear_output()

        if timed_question_count >= timed_question_total:
            display(HTML(f"<h3>🎉 Timed Challenge Complete!</h3>"))
            display(HTML(f"<p><b>Total Score:</b> {timed_question_score} XP</p>"))
            update_next_action_options("Timed Question")
            display(next_action_radio, next_button)
            return

        selected_chapter = chapter_dropdown.value
        questions = chapter_questions.get(selected_chapter, [])
        answers = chapter_answers.get(selected_chapter, [])

        if not questions or not answers:
            display(HTML("<h3 style='color:red;'>⚠️ No questions found for this chapter.</h3>"))
            return

        idx = random.randint(0, len(questions) - 1)
        question = questions[idx]
        correct = answers[idx]
        options = generate_generic_mcq_options(question, correct)

        display(HTML(f"<h3>🕐 Timed Challenge: Question {timed_question_count + 1} of {timed_question_total}</h3>"))
        display(HTML(f"<b>Question:</b> {question}"))
        display(timer_label)

        result_display = widgets.HTML()
        next_btn = widgets.Button(description="➡️ Next", button_style="success", disabled=True)
        submit_btn = widgets.Button(description="Submit Answer", button_style="primary", disabled=True)

        selected_option = {"value": None}

        def stop_timer():
            global timer_running
            timer_running = False
            timer_stop_event.set()

        def disable_buttons():
            for btn in option_buttons:
                btn.disabled = True

        def handle_submit(_):
            global timed_question_score
            if not selected_option["value"]:
                result_display.value = "<p style='color:red;'>⚠️ Please select an option first.</p>"
                return

            stop_timer()
            disable_buttons()

            user_answer = selected_option["value"].strip().lower()
            correct_answer = correct.strip().lower()

            if user_answer == correct_answer:
                award_xp("🕐 Timed Question")
                timed_question_score += 15
                result_display.value = "<div style='color:green; font-weight:bold;'>✅ Correct! +15 XP</div>"
            else:
                result_display.value = f"<div style='color:red; font-weight:bold;'>❌ Incorrect. Correct answer: {correct}</div>"

            next_btn.disabled = False

        submit_btn.on_click(handle_submit)

        option_buttons = []

        def on_click_handler(opt_text, btn_widget):
            def inner(_):
                for btn in option_buttons:
                    btn.button_style = ''  # reset all
                btn_widget.button_style = 'info'  # highlight selected
                selected_option["value"] = opt_text
                submit_btn.disabled = False
            return inner

        for i, option in enumerate(options):
            letter = chr(65 + i)
            btn = widgets.Button(
                description=f"{letter}. {option}",
                layout=widgets.Layout(width='100%', height='auto'),
                button_style=''
            )
            btn.on_click(on_click_handler(option, btn))
            option_buttons.append(btn)

        display(HTML("<style>button.widget-button { white-space: normal; text-align: left; }</style>"))
        display(widgets.VBox(option_buttons))
        display(submit_btn)
        display(result_display)

        def run_timer():
            global timer_running
            timer_running = True
            remaining = timer_seconds
            while remaining > 0 and timer_running:
                timer_label.value = f"⏱️ Time left: {remaining} seconds"
                time.sleep(1)
                remaining -= 1

            if timer_running:
                timer_label.value = "⏱️ Time's up!"
                disable_buttons()
                result_display.value = f"<p style='color:red; font-weight:bold;'>❌ Time's up! Correct answer: {correct}</p>"
                next_btn.disabled = False
                timer_running = False

        timer_stop_event.clear()
        timer_thread = threading.Thread(target=run_timer)
        timer_thread.start()

        def next_question(_):
            global timed_question_count
            timed_question_count += 1
            show_timed_question()

        next_btn.on_click(next_question)
        display(next_btn)

        display(HTML("<hr><h4>✅ What would you like to do next?</h4>"))
        update_next_action_options("Timed Question")
        display(next_action_radio, next_button)


##scenario based with hints
def generate_dynamic_use_case_scenario(chapter):
    summary = chapter_summaries.get(chapter, "No summary available.")
    questions = chapter_questions.get(chapter, [])
    answers = chapter_answers.get(chapter, [])

    if not questions or not answers:
        return None  # Not enough data

    # Choose a relevant Q&A pair for this scenario
    idx = random.randint(0, min(len(questions), len(answers)) - 1)
    question = questions[idx]
    answer = answers[idx]

    # Define a random actor
    actor_roles = ["data analyst", "IT coordinator", "junior accountant", "BI consultant", "operations lead", "technical intern"]
    actor_names = ["Jordan", "Alex", "Taylor", "Sam", "Jamie", "Morgan"]
    actor = f"{random.choice(actor_names)}, a {random.choice(actor_roles)}"

    # Extract a potential goal from the question
    goal = re.sub(r"(?i)\bwhat is\b|\bhow can\b|\bdescribe\b|\bexplain\b", "", question).strip().capitalize()

    # Generate fake success flow steps (simplified version from answer)
    steps = re.split(r'[.,]', answer)
    steps = [s.strip() for s in steps if s.strip()]
    if len(steps) < 3:
        steps += ["They reviewed documentation", "Consulted with the team", "Tested the feature before presenting"]

    # Generate 3 failure distractors (rephrased wrong actions)
    distractors = [
        "Skipped validation and submitted raw data",
        "Relied only on intuition without reviewing facts",
        "Shared the draft without checking for errors",
        "Used outdated methods instead of Excel features"
    ]
    correct_option = f"Applied: {answer}"

    options = random.sample(distractors, 3) + [correct_option]
    random.shuffle(options)

    scenario = {
        "title": f"📘 Use Case: Applying {clean_chapter_name(chapter)} in a Real-World Setting",
        "actor": actor,
        "goal": goal or f"Apply {clean_chapter_name(chapter)} concepts in a work task",
        "preconditions": [
            "Access to relevant tools/software",
            "Basic understanding of chapter concepts"
        ],
        "success_path": steps[:5],
        "failure_paths": distractors,
        "postconditions": "Successful implementation of knowledge from the chapter",
        "decision_point_question": f"What should {actor.split(',')[0]} do next to achieve their goal?",
        "options": {f"{chr(65+i)}. {opt}": ("✅ Correct!" if opt == correct_option else "❌ Try again.") for i, opt in enumerate(options)},
        "hint": f"💡 Think about what the concept '{clean_chapter_name(chapter)}' is meant to help with."
    }

    return scenario




def start_ai_conversation():
    chat_input = widgets.Text(placeholder="Ask me anything about the chapter or topic...")
    send_btn = widgets.Button(description="Send", button_style="primary")
    chat_output = widgets.Output()

    def send_message(_):
        user_query = chat_input.value
        chat_input.value = ""
        with chat_output:
            display(HTML(f"<p><b>👤 You:</b> {user_query}</p>"))
            response = qa_chain.run(user_query)
            display(HTML(f"<p><b>🤖 AI Tutor:</b> {response}</p>"))

    send_btn.on_click(send_message)

    with output:
        clear_output()
        display(HTML("<h3>💬 Chat with AI Tutor</h3>"))
        display(chat_input, send_btn)
        display(chat_output)
        display(HTML("<p style='color:gray;'>Type your question and hit Send. You can ask about concepts, topics, or get help with challenges.</p>"))
        display(HTML("<br><b>🔙 Click below to return:</b>"))
        display(next_action_radio, next_button)






# Shuffle flashcards
flip_button.on_click(flip_flashcard)
next_card_btn.on_click(next_flashcard)
restart_btn.on_click(restart_flashcards)
correct_btn.on_click(mark_correct)
incorrect_btn.on_click(mark_incorrect)


# 🎮 Global XP & Level State
game_state = {
    "xp": 0,
    "level": 1,
    "xp_to_next": 50,
    "history": [],
    "milestones": []

}

# 🏅 Badge Definitions
badge_rules = {
    "Starter": lambda state: state["xp"] >= 10,
    "Level 5 Achiever": lambda state: state["level"] >= 5,
    "Quiz Master": lambda state: "✅ MCQ Quiz" in "".join(state["history"]),
    "Explorer": lambda state: len(set(
        k for k in xp_per_challenge if any(k in h for h in state["history"])
    )) >= 4,
    "Perfect Match": lambda state: any("🧩 Match the Answers" in h for h in state["history"]),
    "Rapid Learner": lambda state: any("🕐 Timed Question" in h for h in state["history"]),
}

# 🎖️ Store earned badges
game_state["badges"] = set()
# 📘 Chapter Progress Tracker
game_state["chapters"] = {}  # Example: { "Excel Basics": {xp: 30, challenges: 3, last_mode: "MCQ"} }


# ✅ Fixed XP thresholds per level (Level 1 to Level 10)
level_xp_thresholds = {
    1: 50,
    2: 75,
    3: 100,
    4: 125,
    5: 150,
    6: 175,
    7: 200,
    8: 225,
    9: 250,
    10: 275
}


# 🏆 XP Config per challenge type
xp_per_challenge = {
    "🧠 Flashcards (Flip)": 5,     # Easy
    "✅ MCQ Quiz": 10,             # Medium
    "✏️ Fill in the Blank": 10,    # Medium
    "🧩 Match the Answers": 12,    # Medium+
    "🕐 Timed Question": 15,       # Hard
    "📘 Scenario-Based (with Hint)": 15  # Hard
}




def show_level_up_celebration(level):
    display(HTML(f"""
        <div style='
            padding: 20px; 
            background-color: #e6ffe6; 
            border-left: 6px solid #4caf50; 
            font-size: 18px;
        '>
            🎉 <b>Congratulations!</b> You've reached <b>Level {level}</b>!
            <br>Keep up the great work and earn more XP!
        </div>
        <p style='font-size:24px;'>🎊🎊🎊</p>
    """))



def award_xp(challenge_type):
    gained = xp_per_challenge.get(challenge_type, 5)
    game_state["xp"] += gained
    game_state["history"].append(f"{challenge_type} ✅ +{gained} XP")
        # 🔁 Track chapter-specific progress
    chapter = chapter_dropdown.value
    if chapter not in game_state["chapters"]:
        game_state["chapters"][chapter] = {"xp": 0, "challenges": 0, "last_mode": challenge_type}

    game_state["chapters"][chapter]["xp"] += gained
    game_state["chapters"][chapter]["challenges"] += 1
    game_state["chapters"][chapter]["last_mode"] = challenge_type


    current_level = game_state["level"]
    xp_required = level_xp_thresholds.get(current_level, game_state["xp_to_next"])

    if game_state["xp"] >= xp_required:
        game_state["milestones"].append({
            "level": current_level,
            "xp_reached": game_state["xp"],
            "xp_needed": xp_required,
            "history": game_state["history"][-5:]
        })

        game_state["level"] += 1
        next_level = game_state["level"]

        # 🛑 If no more levels defined, freeze at max level
        if next_level in level_xp_thresholds:
            game_state["xp_to_next"] = level_xp_thresholds[next_level]
        else:
            game_state["xp_to_next"] = 999999  # Freeze or show MAX LEVEL

        game_state["history"].append(f"🏆 Leveled up to Level {next_level}!")
        show_level_up_celebration(next_level)

            # 🏅 Badge awarding logic
    for badge, condition in badge_rules.items():
        if condition(game_state) and badge not in game_state["badges"]:
            game_state["badges"].add(badge)
            display(HTML(f"<p style='color:gold; font-size:16px;'>🏅 New Badge Unlocked: <b>{badge}</b>!</p>"))


def show_progress_dashboard():
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    def draw_progress_ring(level, xp, xp_to_next):
        progress = min(xp / xp_to_next, 1.0)  # Prevent overflow
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.barh(1, 2 * np.pi, left=0, height=0.3, color="#eee")
        ax.barh(1, 2 * np.pi * progress, left=0, height=0.3, color="#4caf50")
        ax.text(0, 0, f'Lvl {level}\n{xp}/{xp_to_next}', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_axis_off()
        plt.title("🎯 Level Progress", fontsize=14)
        plt.tight_layout()
        plt.show()

    def draw_xp_breakdown(xp_breakdown):
        challenge_types = list(xp_breakdown.keys())
        xp_values = list(xp_breakdown.values())

        colors = plt.cm.tab20.colors
        bar_colors = colors[:len(challenge_types)]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(challenge_types, xp_values, color=bar_colors)

        for bar, xp in zip(bars, xp_values):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height() / 2,
                    f'{int(width)} XP', va='center', fontsize=10, color='black')

        ax.set_title("🧩 XP Gained by Challenge Type", fontsize=14)
        ax.set_xlabel("XP Points")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def draw_xp_growth_chart(milestones):
        levels = [m["level"] for m in milestones]
        xps = [m["xp_reached"] for m in milestones]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(levels, xps, marker="o", linestyle="-")
        ax.set_title("📈 XP Growth Over Levels")
        ax.set_xlabel("Level")
        ax.set_ylabel("Total XP at Level Up")
        plt.tight_layout()
        plt.show()

    def draw_clustered_challenge_chart(milestones):
        level_challenges = {}
        for m in milestones:
            level = f"L{m['level']}"
            level_challenges[level] = {}
            for entry in m["history"]:
                for challenge in xp_per_challenge:
                    if challenge in entry:
                        level_challenges[level][challenge] = level_challenges[level].get(challenge, 0) + 1

        if not level_challenges:
            return

        df_levels = pd.DataFrame(level_challenges).fillna(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        df_levels.T.plot(kind='bar', ax=ax)

        ax.set_title("🧠 Challenge Type Distribution Per Level")
        ax.set_ylabel("Count")
        ax.set_xlabel("Challenge Type")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()





    def calculate_xp_breakdown(history):
        xp_breakdown = defaultdict(int)
        for entry in history:
            for key in xp_per_challenge:
                if key in entry:
                    xp_breakdown[key] += xp_per_challenge[key]
        return xp_breakdown

    xp = game_state["xp"]
    level = game_state["level"]
    xp_to_next = level_xp_thresholds.get(level, game_state["xp_to_next"])
    history = game_state["history"]
    xp_breakdown = calculate_xp_breakdown(history)

    with output:
        clear_output()
        display(HTML(f"""
            <h2 style='color:#2e7d32;'>📊 Your XP Dashboard</h2>
            <p><b>Level:</b> {level} &nbsp;&nbsp; <b>Total XP:</b> {xp}/{xp_to_next}</p>
        """))

        draw_progress_ring(level, xp, xp_to_next)
        draw_xp_breakdown(xp_breakdown)

        display(HTML("<h4>📜 Recent Activity</h4>"))
        for entry in history[-5:]:
            display(HTML(f"🔹 {entry}"))

        if game_state.get("milestones"):
            display(HTML("<h3 style='color:#2e7d32;'>📜 Your XP Journey</h3>"))

            for m in game_state["milestones"]:
                display(HTML(f"""
                    <div style='
                        border-left: 5px solid #4caf50;
                        background-color: #f9fff9;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 8px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    '>
                        <h4 style='margin: 0 0 10px; color: #388e3c;'>🏆 Level {m['level']} Unlocked!</h4>
                        <p><b>XP Reached:</b> {m['xp_reached']} &nbsp;&nbsp; <b>Needed:</b> {m['xp_needed']}</p>
                        <p style='margin-top: 10px;'><b>🔥 Challenges Before Level-Up:</b></p>
                        <ul style='margin-left: 20px; color: #2e7d32;'>{''.join(f"<li>✔️ {entry}</li>" for entry in m['history'])}</ul>
                    </div>
                """))

        if game_state.get("badges"):
            badge_list = "".join(f"<li>🏅 {b}</li>" for b in game_state["badges"])
            display(HTML(f"""
                <h3 style='color:gold;'>🏅 Earned Badges</h3>
                <ul>{badge_list}</ul>
            """))


            draw_xp_growth_chart(game_state["milestones"])
            draw_clustered_challenge_chart(game_state["milestones"])


        update_next_action_options("XP Dashboard")
        display(next_action_radio, next_button)


        # 🔁 Global Radio and Button for Navigation
next_action_radio = widgets.RadioButtons(
    options=[],
    description="Next Step:",
    layout=widgets.Layout(width='auto'),
    style={'description_width': 'initial'}
)

next_button = widgets.Button(description="Continue", button_style="success")
#next_button.on_click(handle_next_action_click)




def update_next_action_options(current_mode_label):
    """
    Dynamically update the options of next_action_radio based on challenge type.
    """
    next_action_radio.options = [
        f"🔁 Restart {current_mode_label}",
        "🎯 Try Another Challenge",
        "📖 Explore Another Learning Mode",
        "📚 Choose Another Chapter",
        "💬 Chat with AI Tutor",
        "📊 Check My XP Dashboard",
        "👤 View My Profile",
        "❌ Exit AI Tutor"
    ]


# ✅ Interactive AI Tutor UI
output = widgets.Output()



# ✅ Step 1: User Name Input
name_input = widgets.Text(placeholder="Enter your name here...")
start_button = widgets.Button(description="Start", button_style="success")

# ✅ Step 2: Ask User for Learning Mode (Select a Chapter)
chapter_dropdown = widgets.Dropdown(
    options=["Select a Chapter"] + list(chapter_summaries.keys()),
    description="📚 Chapter:"
)
confirm_button = widgets.Button(description="Confirm", button_style="success")

# ✅ Step 3: Learning Mode Selection
mode_dropdown = widgets.Dropdown(
    options=["Select Mode", "📊 Business Case", "📖 Storytelling", "🎯 Challenges"],
    description="🎓 Mode:"
)
mode_confirm_button = widgets.Button(description="Go!", button_style="success")

def start_ai_tutor(_=None):
    """Step 1: Greets user and asks for their name."""
    with output:
        clear_output()
        display(HTML("<h3>🤖 AI Tutor: <b>Hello Welcome to Interactive learning Iam your AI today Hope you are doing good! What’s your name?</b></h3>"))
        display(name_input, start_button)

def ask_for_chapter(name):
    """Step 2: Asks user what they want to learn today."""
    with output:
        clear_output()
        if not name.strip():
            display(HTML("<p style='color:red;'>⚠️ Please enter your name to continue.</p>"))
            return

        display(HTML(f"<h3>🤖 AI Tutor: <b>Well, hi {name}! 🎉</b></h3>"))
        display(HTML("<p>You are about to have an interactive learning experience! Before we begin, please select a chapter you'd like to learn today.</p>"))
        display(chapter_dropdown, confirm_button)

start_button.on_click(lambda _: ask_for_chapter(name_input.value))



def show_learning_modes(_):
    """Step 3: Show learning mode after selecting a chapter."""
    chapter = chapter_dropdown.value
    with output:
        clear_output()
        if chapter == "Select a Chapter":
            display(HTML("<p style='color:red;'>⚠️ Please select a valid chapter.</p>"))
            return

        display(HTML(f"<h3>🤖 AI Tutor: <b>Great choice! Let's dive into {chapter} 🚀</b></h3>"))
        display(HTML(f"<p>📖 <b>Quick Summary:</b> {chapter_summaries.get(chapter, 'No summary available.')}</p>"))

        # ✅ Fetch and Display Relevant Image Below Summary
        img = fetch_relevant_image(chapter)
        if img:
            display(img)
        else:
            display(HTML("<p>⚠️ No relevant image found.</p>"))

        # ✅ Show Additional Learning Resources
        more_info_html = fetch_more_info(chapter)
        display(HTML(more_info_html))

        # ✅ Ask user for preferred learning mode
        display(HTML("<p><b>How would you like to learn this chapter?</b></p>"))
        display(mode_dropdown, mode_confirm_button)

confirm_button.on_click(show_learning_modes)

def show_scenario_based_with_hint():
    chapter = chapter_dropdown.value
    scenario = generate_dynamic_use_case_scenario(chapter)

    with output:
        clear_output()
        if not scenario:
            display(HTML("<p style='color:red;'>⚠️ Not enough data to generate a scenario for this chapter.</p>"))
            return

        display(HTML(f"<h3>{scenario['title']}</h3>"))
        display(HTML(f"<b>👤 Actor:</b> {scenario['actor']}<br>"))
        display(HTML(f"<b>🎯 Goal:</b> {scenario['goal']}<br>"))
        display(HTML("<b>✅ Preconditions:</b><ul>" + "".join(f"<li>{p}</li>" for p in scenario["preconditions"]) + "</ul>"))
        display(HTML("<b>📈 Success Path:</b><ol>" + "".join(f"<li>{s}</li>" for s in scenario["success_path"]) + "</ol>"))
        display(HTML("<b>⚠️ Failure Possibilities:</b><ul>" + "".join(f"<li>{f}</li>" for f in scenario["failure_paths"]) + "</ul>"))
        display(HTML(f"<b>🎯 Success Outcome:</b> {scenario['postconditions']}"))

        # Question & Choices
        dropdown = widgets.Dropdown(
            options=["Select"] + list(scenario["options"].keys()),
            description="Your Choice:"
        )
        submit_btn = widgets.Button(description="Submit", button_style="success")
        hint_btn = widgets.Button(description="Hint", button_style="info")
        feedback = widgets.Label()

        def handle_submission(_):
            selection = dropdown.value
            feedback.value = scenario["options"].get(selection, "⚠️ Please choose an option.")
            if "✅" in feedback.value:
                award_xp("📘 Scenario-Based (with Hint)")

        submit_btn.on_click(handle_submission)
        hint_btn.on_click(lambda _: setattr(feedback, 'value', scenario["hint"]))

        display(HTML(f"<h4>📘 Decision Point:</h4><p>{scenario['decision_point_question']}</p>"))
        display(dropdown, widgets.HBox([submit_btn, hint_btn]), feedback)

        update_next_action_options("Scenario-Based")
        display(next_action_radio, next_button)



def process_learning_mode(_):
    """Step 4: Show learning content based on selected mode and allow more exploration."""
    mode = mode_dropdown.value
    chapter = chapter_dropdown.value  # Original chapter (with numbers)

    if chapter == "Select a Chapter":
        with output:
            clear_output()
            display(HTML("<p style='color:red;'>⚠️ Please select a valid chapter.</p>"))
        return

    cleaned_chapter = clean_chapter_name(chapter)

    with output:
        clear_output()

        if mode == "Select Mode":
            display(HTML("<p style='color:red;'>⚠️ Please select a learning mode.</p>"))
            return

        display(HTML(f"<h3 style='color: darkblue;'>🤖 AI Tutor: <b>Fantastic! Let's explore {mode} for {cleaned_chapter} 🔥</b></h3>"))

        if mode == "📊 Business Case":
            try:
                business_case = generate_business_case_logic_v9(cleaned_chapter)
                financials = business_case.get("Financials & Timeline", "No financial data available.").replace("$", "💲")

                formatted_case = f"""
                    <h3 style='color:blue;'>📊 Business Scenario:</h3>
                    <p>{business_case.get("Executive Summary", "")}</p>
                    <h3 style='color:blue;'>📝 Task:</h3>
                    <p>Analyze how <b>{cleaned_chapter}</b> was used to improve business decision-making.</p>
                    <h3 style='color:blue;'>📌 Problem Statement:</h3>
                    <p>{business_case.get("Problem Statement", "")}</p>
                    <h3 style='color:blue;'>⚠️ Why is This Important?</h3>
                    <p>{business_case.get("Why is This Important?", "")}</p>
                    <h3 style='color:blue;'>💡 Proposed Solution:</h3>
                    <p>{business_case.get("Proposed Solution", "")}</p>
                    <h3 style='color:blue;'>🎯 Business Objectives:</h3>
                    <ul>
                        {''.join(f"<li>{obj}</li>" for obj in business_case.get("Business Objectives", []))}
                    </ul>
                    <h3 style='color:blue;'>📊 Scope & Impact:</h3>
                    <p>{business_case.get("Scope & Impact", "")}</p>
                    <h3 style='color:blue;'>💰 Financials & Timeline:</h3>
                    <p>{financials}</p>
                    <h3 style='color:blue;'>🚀 Conclusion & Next Steps:</h3>
                    <p>{business_case.get("Conclusion & Next Steps", "")}</p>
                """
                display(HTML(formatted_case))
            except Exception as e:
                display(HTML(f"<p style='color:red;'>⚠️ Error generating business case: {e}</p>"))

        elif mode == "📖 Storytelling":
            try:
                story = generate_dynamic_storytelling(cleaned_chapter)

                if not story:
                    raise ValueError("⚠️ No valid storytelling content generated.")

                display(HTML(f"<h2 style='color: darkblue;'>📖 Storytelling</h2><p style='color: darkgreen;'>{story}</p>"))

            except Exception as e:
                display(HTML(f"<p style='color:red;'>⚠️ Error generating storytelling: {e}</p>"))
        elif mode == "🎯 Challenges":
            try:
                display(HTML("<h3>🎯 Choose Your Challenge Type:</h3>"))
                challenge_type_dropdown = widgets.Dropdown(
                    options=[
                        "🧠 Flashcards (Flip)",
                        "✅ MCQ Quiz",
                        "✏️ Fill in the Blank",
                        "🧩 Match the Answers",
                        "🕐 Timed Question",
                        "📘 Scenario-Based (with Hint)"
                    ],
                    description="Mode:"
                )
                go_btn = widgets.Button(description="Start", button_style="success")

                def show_selected_challenge(_):
                    challenge_type = challenge_type_dropdown.value
                    clear_output(wait=True)
                    award_xp(challenge_type)
                    display(HTML(f"<h3>🎯 You selected: {challenge_type}</h3>"))

                    selected_chapter = chapter_dropdown.value

                    if challenge_type == "🧠 Flashcards (Flip)":
                        questions = chapter_questions.get(selected_chapter, [])
                        answers = chapter_answers.get(selected_chapter, [])
                        global flashcards, current_index
                        flashcards = list(zip(questions, answers))
                        random.shuffle(flashcards)
                        current_index = 0
                        show_flashcard()

                    elif challenge_type == "✅ MCQ Quiz":

                        global mcq_questions_asked, mcq_score
                        mcq_questions_asked = 0
                        mcq_score = 0
                        show_mcq_question()





                    elif challenge_type == "✏️ Fill in the Blank":

                        start_interactive_fill_blank_game()

                    elif challenge_type == "🧩 Match the Answers":
                        show_matching_game()
                    elif challenge_type == "🕐 Timed Question":
                        show_timed_question()
                    elif challenge_type == "📘 Scenario-Based (with Hint)":
                        show_scenario_based_with_hint()
                    else:
                        display(widgets.Label(value="❌ Please select a valid challenge."))

                go_btn.on_click(show_selected_challenge)
                display(challenge_type_dropdown, go_btn)

            except Exception as e:
                display(HTML(f"<p style='color:red;'>⚠️ Error displaying challenges: {e}</p>"))

#                 # ✅ Show More Info & Next Steps
#                                    # ✅ Let the user decide what to do next
#                 display(HTML("<h3>✅ What would you like to do next?</h3>"))

#                 next_action = widgets.RadioButtons(
#                     options=[
#                         "📖 Explore another learning mode for this chapter",
#                         "📚 Choose a different chapter",
#                         "❌ Exit AI Tutor"
#                     ],
#                     description="🔍 Options:"
#                 )

#                 next_button = widgets.Button(description="Proceed", button_style="success")


        # ✅ Show Additional Learning Resources
        more_info_html = fetch_more_info(cleaned_chapter)
        display(HTML(more_info_html))

        # ✅ Let the user decide what to do next
        display(HTML("<h3>✅ What would you like to do next?</h3>"))

        # ✅ Use the global radio + button
        update_next_action_options(mode)
        display(next_action_radio, next_button)





                # ✅ End of all mode conditionals (after "📖 Storytelling", "🎯 Challenges", etc.)

def show_profile_page():
        from IPython.display import HTML
        import matplotlib.pyplot as plt

        name = name_input.value or "Learner"
        level = game_state["level"]
        xp = game_state["xp"]
        xp_to_next = level_xp_thresholds.get(level, game_state["xp_to_next"])
        badges = list(game_state.get("badges", []))
        avatar_url = "https://api.dicebear.com/7.x/bottts/svg?seed=" + name.replace(" ", "+")  # Random fun avatar

        with output:
            clear_output()
            display(HTML(f"""
                <div style="display:flex; align-items:center; gap:20px;">
                    <img src="{avatar_url}" width="100" height="100" style="border-radius:50%;" />
                    <div>
                        <h2 style="margin-bottom: 5px;">👤 {name}'s Profile</h2>
                        <p><b>Level:</b> {level} | <b>XP:</b> {xp} / {xp_to_next}</p>
                        <p><b>Badges:</b> {', '.join(['🏅 ' + b for b in badges]) if badges else 'None yet'}</p>
                    </div>
                </div>
                <hr style="margin: 15px 0;">
            """))

            # 🧭 Chapter Summary Table
            if game_state["chapters"]:
                display(HTML("<h3>📘 Chapter Progress</h3>"))

                table_html = """
                    <table style="width:100%; border-collapse:collapse;">
                        <tr style="background:#f0f0f0;">
                            <th style="padding:8px;">Chapter</th>
                            <th style="padding:8px;">XP</th>
                            <th style="padding:8px;">Challenges</th>
                            <th style="padding:8px;">Last Mode</th>
                        </tr>
                """
                for chapter, stats in game_state["chapters"].items():
                    table_html += f"""
                        <tr>
                            <td style="padding:8px;">{chapter}</td>
                            <td style="padding:8px;">{stats['xp']}</td>
                            <td style="padding:8px;">{stats['challenges']}</td>
                            <td style="padding:8px;">{stats['last_mode']}</td>
                        </tr>
                    """
                table_html += "</table>"
                display(HTML(table_html))
            else:
                display(HTML("<p>No chapter activity yet. Start learning!</p>"))

            update_next_action_options("Profile")
            display(next_action_radio, next_button)


def handle_next_action_click(_):
    with output:
        clear_output(wait=True)
        selection = next_action_radio.value

        if selection.startswith("🔁 Restart Flashcards"):
            restart_flashcards(None)

        elif selection.startswith("🔁 Restart MCQs"):
            global mcq_questions_asked, mcq_score
            mcq_questions_asked = 0
            mcq_score = 0
            show_mcq_question()

        elif selection.startswith("🔁 Restart Fill-in-the-Blank"):
            start_interactive_fill_blank_game()

        elif selection.startswith("🔁 Restart Match the Answers"):
            show_matching_game()

        elif selection.startswith("🔁 Restart Timed Question"):
            global timed_question_count, timed_question_score
            timed_question_count = 0
            timed_question_score = 0
            show_timed_question()

        elif selection.startswith("🔁 Restart Scenario-Based"):
            show_scenario_based_with_hint()

        elif selection == "🎯 Try Another Challenge":
            mode_dropdown.value = "🎯 Challenges"
            process_learning_mode(None)

        elif selection == "📖 Explore Another Learning Mode":
            display(mode_dropdown, mode_confirm_button)

        elif selection == "📚 Choose Another Chapter":
            display(chapter_dropdown, confirm_button)

        elif selection == "📊 Check My XP Dashboard":
            show_progress_dashboard()

        elif selection == "👤 View My Profile":
            show_profile_page()


        elif selection == "💬 Chat with AI Tutor":
            start_ai_conversation()

        elif selection == "❌ Exit AI Tutor":
            display(HTML("<p>👋 Thanks for learning with AI Tutor! Stay curious! 🚀</p>"))

# ✅ Bind Next Button to Handler
next_button.on_click(handle_next_action_click)




# === FINAL EVENT BINDINGS ===

confirm_button.on_click(show_learning_modes)
mode_confirm_button.on_click(process_learning_mode)



# Start AI tutor UI
start_ai_tutor()

# Bind dropdown button to function
mode_confirm_button.on_click(process_learning_mode)

# Show the output interface
display(output)

