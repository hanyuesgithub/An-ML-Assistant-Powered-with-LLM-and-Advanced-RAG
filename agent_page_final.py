import asyncio
import streamlit as st
import requests

# 设置页面标题和图标
st.set_page_config(page_title="AI助手对话", page_icon=":robot_face:",
                       layout="wide",  # 页面布局
    initial_sidebar_state="expanded",  # 侧边栏初始状态
    )

from streamlit_chat import message
# 设置页面背景颜色和字体
from openai import OpenAI
import streamlit as st

image = {
    "user":"User.png",
    "assistant":"assistant_big.png"
}

def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"], avatar=image[message["role"]]):
            st.markdown(message["content"])

def add_user_message_to_session(prompt):
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        # with st.chat_message("user", avatar="User.png"):
        with st.chat_message("user", avatar=image["user"]):
            st.markdown(prompt)


# 定义API的URL
# context_api_url = "http://36.212.25.245:5117/get_context"
context_api_url = "http://36.212.25.245:5117/get_context_combined"
answer_api_url = "http://36.212.25.245:5115/generate_answer"


# def gain_results(query, context):
#     # 发送POST请求获取context
#     context_response = requests.post(context_api_url, json={"input": query})

#     if context_response.status_code == 200:
#         context_result = context_response.json()
#         context = context_result["context"]
#         print("Retrieved Context:", context)

#         # 准备输入数据,包括question和context
#         input_data = {
#             "question": query,
#             "context": context
#         }

#         # 发送POST请求生成答案
#         answer_response = requests.post(answer_api_url, json=input_data)

#         if answer_response.status_code == 200:
#             answer_result = answer_response.json()
#             generated_answer = answer_result["answer"]
#             print("Generated Answer:", generated_answer)

#             with st.chat_message("assistant", avatar=image["assistant"]):
#                 st.markdown(generated_answer)

#             st.session_state["messages"].append(
#                 {"role": "assistant", "content": generated_answer}
#             )
#             return generated_answer
#         else:
#             print("Error generating answer:", answer_response.status_code)
#             return "抱歉,生成答案时出错。"
#     else:
#         print("Error retrieving context:", context_response.status_code)
#         return "抱歉,获取上下文时出错。"

async def gain_context(query):
      # 发送POST请求获取context
    context_response = requests.post(context_api_url, json={"input": query})

    if context_response.status_code == 200:
        context_result = context_response.json()
        context = context_result["context"]
        print("Retrieved Context:", context)

        # # 准备输入数据,包括question和context
        # input_data = {
        #     "question": query,
        #     "context": context
        # }
    return context

async def gain_result(query, context):
    input_data = {
            "question": query,
            "context": context
        }
    
    print("input_data!!!!", input_data)
    answer_response = requests.post(answer_api_url, json=input_data)
    print("answer_response!!!!", answer_response)
    if answer_response.status_code == 200:
        answer_result = answer_response.json()
        generated_answer = answer_result["answer"]
        print("Generated Answer:", generated_answer)

        # with st.chat_message("assistant", avatar=image["assistant"]):
        #     st.markdown(generated_answer)

        # st.session_state["messages"].append(
        #     {"role": "assistant", "content": generated_answer}
        # )
        return generated_answer
    else:
        print("Error generating answer:", answer_response.status_code)
        return "抱歉,生成答案时出错。"

    
async def generate_assistant_response(query):
    # add_user_message_to_session 显示消息的时候做了处理，所以这里不需要再次添加最新提问
    print('history-->')
    history = st.session_state["messages"]
    print(history)
    with st.chat_message("assistant", avatar=image["assistant"]):
        message_placeholder = st.empty()
        
        context = await gain_context(query)
        returned = ""+context
        st.markdown(f'<p style="color: black; font-size: 15px;">{returned}</p>', unsafe_allow_html=True)
        result = await gain_result(query, context)
        full_response = result
        
        message_placeholder.markdown(full_response)
        
        # for response in client.chat.completions.create(
        #         model=model,
        #         temperature=0,
        #         messages=history,
        #         stream=True,
        # ):
        #     try:
        #         full_response += response.choices[0].delta.content
        #     except Exception as e:
        #         print("")
        #     message_placeholder.markdown(full_response + "▌")
        
        # message_placeholder.markdown(full_response)
        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )
    return full_response

def hide_streamlit_header_footer():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

# def main():
#         # 显示标题和描述
#     col1, col2, col3 = st.columns(3)
#     with col2:
#         st.title("mechine learning assistant")
#         st.image("scifi_girl.png", width=120)

#     hide_streamlit_header_footer()
#     display_existing_messages()

#     query = st.chat_input("you can ask me any question")

#     if query:
#         print(query)
#         add_user_message_to_session(query)
#         response = generate_assistant_response(query)
#         print(response)

# if __name__ == "__main__":
#     main()
async def main():
    col1, col2, col3 = st.columns([2.35, 2, 2])
    with col2:
        st.title("ML assistant")
        st.image("assistant.png", width=200)

    hide_streamlit_header_footer()
    display_existing_messages()

    query = st.chat_input("you can ask me any question")

    if query:
        print(query)
        add_user_message_to_session(query)
        response = await generate_assistant_response(query)
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
