
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from konlpy.tag import Twitter
#from ckonlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
import jpype


# JVM 경로 지정
jvm_path = 'C:/Program Files/Java/jre-1.8/bin/server/jvm.dll'






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
        my_new_words = [('저장소', 'Noun'), ('취급소', 'Noun'), ('제조소', 'Noun')]
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
    st.title("위험물안전관리법 가이드 챗봇 🤖")
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
    st.title("위험물안전관리법 가이드 챗봇 🤖")
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

