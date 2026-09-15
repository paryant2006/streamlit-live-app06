from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------
# 1. Create the LLM
# -----------------------------------------

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# -----------------------------------------
# 2. Create Role + Few-Shot Prompt
# -----------------------------------------

messages = [

    # ROLE
    SystemMessage(
        content="""
You are a friendly Nutrition Assistant.

Your job is to help users understand whether
a food choice is generally healthy.

Classify food into one of these categories:

Healthy
Moderate
Avoid

Always respond using this format:

Category: <category>

Reason:
<short explanation>

Suggestion:
<short suggestion>

Keep your answer simple and suitable for beginners.
"""
    ),

    # -----------------------------------------
    # FEW-SHOT EXAMPLE 1
    # -----------------------------------------

    HumanMessage(
        content="I want to eat an apple."
    ),

    AIMessage(
        content="""
Category: Healthy

Reason:
Apple contains fiber, vitamins and natural nutrients.

Suggestion:
Apple can be included as part of a balanced diet.
"""
    ),

    # -----------------------------------------
    # FEW-SHOT EXAMPLE 2
    # -----------------------------------------

    HumanMessage(
        content="I want to eat pizza."
    ),

    AIMessage(
        content="""
Category: Moderate

Reason:
Pizza can contain a lot of refined carbohydrates,
cheese, salt and calories depending on the preparation.

Suggestion:
Eat it occasionally and prefer vegetables and
moderate cheese as toppings.
"""
    ),

    # -----------------------------------------
    # FEW-SHOT EXAMPLE 3
    # -----------------------------------------

    HumanMessage(
        content="I want to drink sugary soda every day."
    ),

    AIMessage(
        content="""
Category: Avoid

Reason:
Sugary drinks can contain a large amount of
added sugar and provide little nutritional value.

Suggestion:
Prefer water or unsweetened drinks for regular use.
"""
    )
]


# -----------------------------------------
# 3. Streamlit UI
# -----------------------------------------

st.title("🥗 Your Personal Assistant for Healthy Food")

food = st.chat_input(
    "What food or drink are you thinking about?"
)

if food:

    st.chat_message("user").write(food)

    # Add the REAL question
    messages.append(
        HumanMessage(
            content=f"I want to eat or drink {food}."
        )
    )

    # -----------------------------------------
    # 4. Send everything to Groq
    # -----------------------------------------

    with st.spinner("Thinking..."):
        response = model.invoke(messages)

    # -----------------------------------------
    # 5. Display result
    # -----------------------------------------

    st.chat_message("assistant").write(response.content)
