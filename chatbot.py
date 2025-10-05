# %%
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("gpt-4o", model_provider="openai")

from langchain_core.messages import HumanMessage, SystemMessage

print(model.invoke([HumanMessage(content="Hi! I'm Bob")]))
print(model.invoke([HumanMessage(content="What's my name?")]))
# %%
from langchain_core.messages import AIMessage

model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)

# %%
