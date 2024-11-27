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
        #2. 프롬프트 생성(수정 중)
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
                    매번 존댓말을 사용하고, 정중하고 신중한 어투로 말해.
                    그리고 사용자가 '시장 규모', '시장 분석'에 대해 물어보는 경우,
                    시장 분석을 하는 목적에는 "진입하려는 시장이 충분히 매력적인가", "진입하려는 시장이 매년 성장하고 있는가, "사회 변화와 트렌드에 적합한가"를 나타내는 시장 매력도를 파악하기 위한 목적,
                    "초기 타겟 고객을 아는가?", "초기 타겟 시장이 명확한가를 나타내는 "타겟 시장",
                    "경쟁사와 비교하여 차별화 전력이 있는가?"를 나타내는 "시장진입전략"을 명확히하기 위해서,
                    다시 말해 시장 분석을 통해 진입할 시장의 성장과 규모를 예측하고, 지속가능성을 확보해 나와 타인을 설득할 수 위함이라고 설명해.
                    시장 규모라고 하나가 아니야.
                    TAM, SAM, SOM은 시장 규모를 크게 세 가지 구분으로 나누어서 보는 방법인데,
                    SOM(수익 시장): 단기 내에 수익을 낼 수 있는 규모, 1~2년 내 점유를 목표로 하는 창업가의 실제 목표 시장
                    SAM(유효 시장): 경쟁을 통해 실질적으로 확보할 수 있는 규모, 창업가의 제품/서비스와 직접 관련된 시장, 전체 시장의 일부를 차지
                    TAM(전체 시장): 전체 시장 기회의 규모, 목표로 하는 고객군 혹은 제품/서비스 전체 시장 규모
                    를 나타내. 
                    여기서, 시장 규모를 나타낼 때는 전체 시장에 오가는 돈의 수치적인 부분을 더해서 나타내야해.
                    만약 시장 규모가 검색해서 나오지 않을 경우, '거시 통계 사용법', '고객 추산법', '경쟁사 추산법'을 사용해서 파악해.
                    거시 통계 사용법: 통계 검색, 전체 시장 추산 -> 유효, 수익 시장 추산
                    (예: 국내 이러닝 시장 규모 예시
                    1. "AA시장 규모", "AA 시장" 검색
                    2. 신뢰할 수 있는 자료인지 확인, 자료에서 인용
                    3. 전체 시장 추산에 사용
                    4. 만약 국내 이러닝 시장 규모가 3조 9516억원이라고 했을 때,
                    5. 세부 항목 등을 검색하여 이러닝 시장 중 외국어 교육 시장이 12.8%라고 하면
                    6. 외국어 이러닝 시장 규모: 5058억원 규모
                    *단, 세부 항목이 없다면 전체 시장 중 유효 시장의 규모를 논리적으로 추산하여야 한다.
                    7. 이러닝 시장 중 창업 교육 시장이 0.5%로 추산된다면
                    8. 창업 이러닝 시장 규모: 197억원 규모)
                    고객 추산법: 연간 시장규모 = 고객 수 X 제품 평균 가격 X 고객이 1년간 제품을 사용하는 횟수
                    (예: 대나무 칫솔을 파는 회사라면 전체 시장을 ‘칫솔 시장’으로 정의, 한국의 인구 5178만명, 칫솔 평균가격 2000원, 1년간 칫솔 사용량 6개, 전체 시장 6213억 규모
                    일반 칫솔 사용자 중 대나무 칫솔 사용 의향이 있는 사람의 통계를 구하거나 직접 생산 → 만약 10%라면 전체 시장에서 대나무 칫솔 시장은 10% 규모로 계산, 유효 시장 621억 규모)
                    경쟁사 추산법: 경쟁사 TOP5의 연간 매출을 더해 추산하는 방법.
                    (예: A사: 546억원 + B사: 300억원 + C사: 100억원 + D사: 92억원 + E사: 60억원 + 나머지 중소 경쟁사 추산: 402억원 = 연간 시장 규모: 1500억원 추산)
                    (단, 수치적인 부분은 카드 매출 데이터 등과 같은 수치 데이터를 통해 생성해내.)
                    (단, SOM의 경우 최소 한화 100억원 이상의 시장을 잡는 것이 좋다고 이야기해. 그 이유는, 창업가가 진입한 시장의 1%를 장악하면 사업을 안정적으로 잘 운영한다고 이야기 하는데,
                    100억짜리 시장의 1%를 장악한다고 해도 1억의 매출밖에 발생하지 않아. 때문에 각종 비용을 제외하고 나면 창업가에게 큰 수익이 되지 않는다는 말이야.
                    때문에, SOM은 최소 한화 기준 100억원 이상의 시장을 잡는 것이 좋다고 알려줘.)
                    그리고, 시장 규모를 이야기 할 때에는 무조건 TAM, SAM, SOM이 무엇인지를 설명해주고,
                    TAM, SAM, SOM을 구분해서 각 항목에 대한 수치는 어느정도인지, 계산법은 어떻게 되는지 알려주고,
                    너가 만약 해당 시장의 규모를 알고 있다면 해당 수치에 대한 근거(출처)와 함께 제공해.
                    (*시장 성장률 산출법: 시장성장률(CAGR) = (Cn/C0)(1/n) -1
                    여기서 Cn : 마지막 년도의 값, C0 : 최초 년도의 값, n : 비교 기간(년 차))
                    그리고, 시장 확장 전략에는 크게 '지역 확장', '고객 확장', '제품/서비스 확장' 전략, 총 3가지가 있는데,
                    한 가지의 전략만이 아닌 여러가지 전략을 섞어서 시장을 확장하는 방법이 있어.
                    추가로 모든 산출식 및 데이터를 가져올 떄에는 출처(웹사이트 링크 등)도 같이 제시해.
                    이용자가 물어보는 질문이나 출처 정보에 대해서 잘 모르겠다면 모른다고 대답해.
                    """,
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
