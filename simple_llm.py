# %%
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("gpt-4o", model_provider="openai")

# %%
messages = [
    SystemMessage(content="Translate the following from English into Japanese."),
    HumanMessage(content="hi!"),
]

model.invoke(messages)

# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
print("is runnable: ", isinstance(prompt_template, Runnable)) # prompt_template is a Runnable
prompt: ChatPromptTemplate = prompt_template.format_prompt(language="Japanese", text="hi!")
model.invoke(prompt.to_messages())
# %%

# これも同じ
chain = prompt_template | model
chain.invoke({"language": "Japanese", "text": "hi!"})

# %%
