import streamlit as st
from utils import print_messages, StreamHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

st.set_page_config(page_title="AI 창업 어시스턴트", page_icon="😎")
st.title("😎AI 창업 어시스턴트😎")

# OpenAI API 키 가져오기
# # .env 파일 로드
# load_dotenv()

# # API 키 설정
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     st.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
# os.environ["OPENAI_API_KEY"] = api_key


if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 채팅 대화 기록을 저장하는 store 세션 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

with st.sidebar:
    session_id = st.text_input("Session Id", value="sample_id")
    # 대화기록 초기화 버튼
    clear_btn = st.button("대화기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        #대화 기록 초기화
        st.session_state["store"] = dict()
        #새로고침
        #st.experimental_rerun()

#print_messages()함수 사용: 이전 대화 기록을 출력해주는 함수
print_messages()

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]: # 세션 ID가 store에 없는 경우
        #새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids] #해당 세션 ID에 대한 세션 기록 반환

if user_input := st.chat_input("궁금한 것을 입력하세요."):
    #사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    #AI의 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        #1. 모델 생성
        llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, callbacks=[stream_handler])
        #2. 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """스타트업 씬에서 창업가는 약자에 속해.
                    이런 약자들을 "언더독"이라고 부르곤 해.
                    스타트업 씬에서 "언더독"에 속하는 창업가들을 도와 "탑독"이 되기를 바라는 마음에서
                    너의 이름은 "탑독"이야.
                    너는 진중하고, 신중한 성격을 보유한 성격의 연쇄창업가야. 
                    대기업에 너가 만든 회사를 매각한 경험도 보유하고 있는 성공한 창업가야. 
                    너의 창업 경험을 바탕으로 예비/초기 단계의 창업가들에게 코칭을 하는 창업 코치로써 조언을 해줘야해.
                    너한테 질문을 하는 창업가들은 사업에 대한 실질적인 조언과 더불어,
                    사업계획서(Ms PowerPoint로 작성된 발표용, MS Word로 작성된 제출용) 작성에 대한 조언을 듣고싶어해.
                    단, 너가 코칭할 교육생들은 모두 정부지원사업 합격과 민간 투자 유치를 목표로하는 사업가들이야.
                    매번 존댓말을 사용하고, 정중하고 신중한 어투로 말해.""",
                ),
                #대화 기록을 변수로 사용, history가 MessageHistory의 key가 됨
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"), #사용자의 질문을 입력
            ]
        )
        chain = prompt | llm

        chain_with_memory = RunnableWithMessageHistory( #RunnableWithMessageHistory 객체 생성
            chain, #실행할 Runnable 객체
            get_session_history, # 세션 기록을 가져오는 함수
            input_messages_key="question", #사용자 질문의 키
            history_messages_key="history", #기록 메시지의 키
        )

        response = chain_with_memory.invoke(
            {"question" : user_input},
            # 세션 ID 설정
            config={"configurable": {"session_id": session_id}}
        )
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response.content)
        )
