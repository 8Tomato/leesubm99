
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from konlpy.tag import Twitter
#from ckonlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu

# 새로 추가한 사이드바 메뉴
with st.sidebar:
    choose = option_menu("메뉴", ["질문란", "위험물 지정수량"],
                         icons=['chat-right-text-fill', 'kanban'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#7ABA78"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "21px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#0A6847"},
    }
    )

# "질의응답" 옵션일 때 실행되는 기능
if choose == "질문란":
    

    # 피클 파일 로드 함수
    @st.cache_resource
    def load_chatbot_model(model_path):
        with open(model_path, 'rb') as f:
            my_vectorizer = pickle.load(f)
            questions_processed = pickle.load(f)
            answers = pickle.load(f)
        X = my_vectorizer.transform(questions_processed).toarray()  # X 생성
        return my_vectorizer, questions_processed, answers, X

    # 한글 전처리 함수
    def korean_processor(lines):
        tagger = Twitter()
        my_new_words = [   ('저장소', 'Noun'),
                        ('취급소', 'Noun'),
                        ('제조소', 'Noun'),
                        ('위치', 'Noun'),
                        ('구조', 'Noun'),
                        ('설비', 'Noun'),
                        ('기술기준', 'Noun'),
                        ('대상', 'Noun'),
                        ('지정수량', 'Noun'),
                        ('구분', 'Noun'),
                        ('출입', 'Noun'),
                        ('검사', 'Noun'),
                        ('권한', 'Noun'),
                        ('사고', 'Noun'),
                        ('피해', 'Noun'),
                        ('조사', 'Noun'),
                        ('소방청장', 'Noun'),
                        ('군부대', 'Noun'),
                        ('조사위원회', 'Noun'),
                        ('설치자', 'Noun'),
                        ('완공검사', 'Noun'),
                        ('사유', 'Noun'),
                        ('법률', 'Noun'),
                        ('허가', 'Noun'),
                        ('조치', 'Noun'),
                        ('안전조치', 'Noun'),
                        ('과징금', 'Noun'),
                        ('책임', 'Noun'),
                        ('자격', 'Noun'),
                        ('역할', 'Noun'),
                        ('위험 분석', 'Noun'),
                        ('교육', 'Noun'),
                        ('환경', 'Noun'),
                        ('규정', 'Noun'),
                        ('내용', 'Noun'),
                        ('문제', 'Noun'),
                        ('기준', 'Noun'),
                        ('해임', 'Noun'),
                        ('대행', 'Noun'),
                        ('수행', 'Noun'),
                        ('처리', 'Noun'),
                        ('존중', 'Noun'),
                        ('요건', 'Noun'),
                        ('감독', 'Noun'),
                        ('참여', 'Noun'),
                        ('이동탱크저장소', 'Noun'),
                        ('행정안전부령', 'Noun'),
                        ('운송', 'Noun'),
                        ('시', 'Noun'),
                        ('운반용기', 'Noun'),
                        ('운송책임자', 'Noun'),
                        ('안전사고', 'Noun'),
                        ('종류', 'Noun'),
                        ('지원', 'Noun'),
                        ('방법', 'Noun'),
                        ('목적', 'Noun'),
                        ('대응', 'Noun'),
                        ('절차', 'Noun'),
                        ('용기', 'Noun'),
                        ('주의사항', 'Noun'),
                        ('발생', 'Noun'),
                        ('범위', 'Noun'),
                        ('안전', 'Noun'),
                        ('주의', 'Noun'),
                        ('공무원', 'Noun'),
                        ('공공', 'Noun'),
                        ('사용정지명령', 'Noun'),
                        ('위반', 'Noun'),
                        ('원인', 'Noun'),
                        ('사고조사위원회', 'Noun'),
                        ('구성', 'Noun'),
                        ('탱크시험자', 'Noun'),
                        ('명령', 'Noun'),
                        ('무허가장소', 'Noun'),
                        ('저장', 'Noun'),
                        ('취급', 'Noun'),
                        ('소', 'Noun'),
                        ('안전관리자', 'Noun'),
                        ('선임', 'Noun'),
                        ('처벌', 'Noun'),
                        ('취급기준', 'Noun'),
                        ('벌금', 'Noun'),
                        ('납부', 'Noun'),
                        ('변경허가', 'Noun'),
                        ('변경', 'Noun'),
                        ('탱크안전성능시험', 'Noun'),
                        ('점검', 'Noun'),
                        ('소방공무원', 'Noun'),
                        ('경찰공무원', 'Noun'),
                        ('위험물안전관리법', 'Noun'),
                        ('소방본부장', 'Noun'),
                        ('소방서장', 'Noun'),
                        ('신원확인', 'Noun'),
                        ('공개시간', 'Noun'),
                        ('관계인', 'Noun'),
                        ('증표', 'Noun'),
                        ('시간', 'Noun'),
                        ('변경공사', 'Noun'),
                        ('사망', 'Noun'),
                        ('양도', 'Noun'),
                        ('사용정지', 'Noun'),
                        ('용도', 'Noun'),
                        ('관계자', 'Noun'),
                        ('대리자', 'Noun'),
                        ('응급', 'Noun'),
                        ('출입통제', 'Noun'),
                        ('신고', 'Noun'),
                        ('소재지', 'Noun'),
                        ('재선임', 'Noun'),
                        ('위험물취급자격자', 'Noun'),
                        ('지시', 'Noun'),
                        ('관련', 'Noun'),
                        ('의견', 'Noun'),
                        ('추진', 'Noun'),
                        ('특별시장', 'Noun'),
                        ('소통', 'Noun'),
                        ('방지', 'Noun'),
                        ('교육수료증', 'Noun'),
                        ("작업자", "Noun"),
                        ("예방", "Noun"),
                        ("대통령령", "Noun"),
                        ("취급자격자", "Noun"),
                        ("전문적", "Noun"),
                        ("지식", "Noun"),
                        ("능력", "Noun"),
                        ("관리", "Noun"),
                        ("화학물질", "Noun"),
                        ("법", "Noun"),
                        ("시설", "Noun"),
                        ("장비", "Noun"),
                        ("훈련", "Noun"),
                        ("유지보수", "Noun"),
                        ("이해도", "Noun"),
                        ("리더십", "Noun"),
                        ("생명", "Noun"),
                        ("분석", "Noun"),
                        ("대응책", "Noun"),
                        ("위험성", "Noun"),
                        ("수칙", "Noun"),
                        ("안전수칙", "Noun"),
                        ("프로그램", "Noun"),
                        ("운송자", "Noun"),
                        ("화재", "Noun"),
                        ("오염", "Noun"),
                        ("인명피해", "Noun"),
                        ("조작", "Noun"),
                        ("보관", "Noun"),
                        ("사용", "Noun"),
                        ('위험물', 'Noun'),
                        ('적재', 'Noun'),
                        ('운반방법', 'Noun'),
                        ('중요기준', 'Noun'),
                        ('세부기준', 'Noun'),
                        ('응급조치', 'Noun'),
                        ('영향', 'Noun'),
                        ('가능성', 'Noun'),
                        ('근거', 'Noun'),
                        ('시도지사', 'Noun'),
                        ('시행', 'Noun'),
                        ('국가기술자격법', 'Noun'),
                        ('운송과정', 'Noun'),
                        ('운송용기', 'Noun'),
                        ('지방', 'Noun'),
                        ('협조', 'Noun'),
                        ('정지명령', 'Noun'),
                        ('제거', 'Noun'),
                        ('긴급', 'Noun'),
                        ('방해', 'Noun'),
                        ('금지', 'Noun'),
                        ('제재', 'Noun'),
                        ('운영', 'Noun'),
                        ('직위', 'Noun'),
                        ('표시', 'Noun'),
                        ('협력', 'Noun'),
                        ('진압대책', 'Noun'),
                        ('정지', 'Noun'),
                        ('요구', 'Noun'),
                        ('자격증', 'Noun'),
                        ('국가기술자격증', 'Noun'),
                        ('전문지식', 'Noun'),
                        ('사람', 'Noun'),
                        ('도지사', 'Noun')  ]
        for a_word, a_pos in my_new_words:
            tagger.add_dictionary(a_word, a_pos)
        my_stopwords = ['이란', '것입', '때', '네', '의', '및', '어떤', '그', '때']
        res = []
        for a_line in lines:
            a_line = tagger.pos(a_line)
            a_line_processed = ''
            for a_word, a_pos in a_line:
                if a_pos in ['Noun', 'Verb', 'Adjective'] and a_word not in my_stopwords:
                    a_line_processed += a_word + ' '
            if len(a_line_processed) > 10:
                res.append(a_line_processed)
        return res

    # 피클 파일 경로
    pickle_path = 'final_project/chatbot_model.pkl'

    # 피클 파일 로드
    my_vectorizer, questions_processed, answers, X = load_chatbot_model(pickle_path)

    # 챗봇 동작 함수
    def chatbot(user_input):
        user_input_processed = korean_processor([user_input])
        user_x = my_vectorizer.transform(user_input_processed).toarray()[0]

        css = []
        for i in range(len(questions_processed)):
            cs = np.dot(X[i], user_x)  
            css.append(cs)

        i_max = np.array(css).argmax()

        return answers[i_max]

    # Streamlit 앱 생성
    st.title("위험물안전관리 가이드 챗봇 🤖")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    # 사용자 입력 받기
    user_input = st.text_input("질문을 입력하세요:")

    st.write("")
    # 챗봇 응답 표시
    if user_input:
        response = chatbot(user_input)
        st.markdown(f"<div class='styled-text'>{response}</div>", unsafe_allow_html=True)
        
        st.markdown("""
    <style>
    .styled-text {
        padding: 10px;
        background-color: #7ABA78;
        border-radius: 5px;
        font-size: 20px;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)
        
    



# "위험물 지정수량" 옵션일 때 실행되는 기능
elif choose == "위험물 지정수량":
    # Streamlit 앱 생성
    st.title("위험물안전관리 가이드 챗봇 🤖")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    
    
    
    # 데이터 불러오기
    df = pd.read_csv('final_project/위험물1.csv', encoding='cp949')

    # 위험물 유별 선택
    language = ['제1류: 산화성고체', '제2류: 가연성고체', '제3류: 자연발화성물질 및 금수성물질', '제4류: 인화성액체', '제5류: 자기반응성물질', '제6류: 산화성액체']
    my_choice = st.selectbox('유별을 선택하세요:', language)

    
    
        # 선택에 따라 데이터프레임 필터링
    if my_choice == language[0]:
        selected_category = '제1류'  # 선택한 유별의 카테고리
    elif my_choice == language[1]:
        selected_category = '제2류'  # 선택한 유별의 카테고리
    elif my_choice == language[2]:
        selected_category = '제3류'  # 선택한 유별의 카테고리
    elif my_choice == language[3]:
        selected_category = '제4류'  # 선택한 유별의 카테고리
    elif my_choice == language[4]:
        selected_category = '제5류'  # 선택한 유별의 카테고리
    elif my_choice == language[5]:
        selected_category = '제6류'  # 선택한 유별의 카테고리

    selected_df = df[df['유별'] == selected_category]  # 선택한 카테고리에 해당하는 행 필터링

    # 선택지 생성
    options = selected_df['품명'].tolist()  # 선택한 카테고리에 속하는 모든 품목을 선택지로 설정
    selected_item = st.selectbox(f"{selected_category} 품목을 선택하세요:", options)  # 품목 선택
    st.write("")
    # 선택한 품목에 대한 지정수량 출력
    selected_quantity = selected_df[selected_df['품명'] == selected_item]['지정수량'].iloc[0]
    #styled_quantity = f"<span style='font-size:25px;'>{selected_quantity}</span>"
    #st.write(f"지정수량: {styled_quantity}", unsafe_allow_html=True)

    # 선택한 품목에 대한 지정수량 출력
    styled_quantity = f"<span style='font-size:25px;'>{selected_quantity}</span>"
    st.markdown(f"<div class='styled-text'>지정수량: {styled_quantity}</div>", unsafe_allow_html=True)


    
        
    st.markdown("""
    <style>
    .styled-text {
        padding: 10px;
        background-color: #7ABA78;
        border-radius: 5px;
        font-size: 20px;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

