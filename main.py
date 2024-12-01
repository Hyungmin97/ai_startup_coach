import streamlit as st
from utils import print_messages, StreamHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json
import glob

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 창업 어시스턴트", page_icon="😎")
st.title("😎AI 창업 어시스턴트😎")

# OpenAI API 키 가져오기
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

# Streamlit 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = dict()

# 사이드바 설정
with st.sidebar:
    session_id = st.text_input("Session Id", value="sample_id")
    clear_btn = st.button("대화기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        st.session_state["store"] = dict()

# 대화 기록 출력
print_messages()

# JSON 파일 로드 함수 (폴더 내 모든 JSON 파일 읽기)
def load_all_chunks(folder_path):
    all_chunks = []
    try:
        for file_path in glob.glob(os.path.join(folder_path, "*.json")):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_chunks.extend(data.get("chunks", []))  # 각 JSON의 "chunks" 키에서 데이터 추출
    except Exception as e:
        st.error(f"JSON 파일 로드 중 오류 발생: {e}")
    return all_chunks

# 프롬프트 파일 로드 함수
def load_prompt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        st.error(f"프롬프트 파일 로드 중 오류 발생: {e}")
        return None

# JSON 데이터 로드 및 가공
chunks_folder = "data/chunks/"  # JSON 파일들이 저장된 폴더 경로
all_chunks_data = load_all_chunks(chunks_folder)

# JSON 데이터를 텍스트로 변환
chunks_context = "\n".join(chunk["content"] for chunk in all_chunks_data if "content" in chunk)

# 프롬프트 데이터 로드
prompt_path = "data/prompts/prompt.txt"  # 프롬프트 파일 경로
prompt_text = load_prompt(prompt_path)
if not prompt_text:
    st.stop()

# 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

# 사용자 입력
if user_input := st.chat_input("궁금한 것을 입력하세요."):
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    # AI 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        # 모델 생성
        llm = ChatOpenAI(model="gpt-4", streaming=True, callbacks=[stream_handler])

        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{prompt_text}\n\n아래는 참고할 데이터입니다:\n{chunks_context}"
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # RunnableWithMessageHistory 설정
        chain = prompt | llm
        chain_with_memory = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        # 사용자 입력 처리 및 AI 응답 생성
        response = chain_with_memory.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response.content)
        )

        #추가 학습 필요 데이터
        #https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15110302#/