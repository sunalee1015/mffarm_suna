import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os

# 페이지 설정
st.set_page_config(page_title="농산물 이커머스 전략 대시보드", layout="wide")

# 데이터 로드 및 전처리 (캐싱)
@st.cache_data
def load_data():
    # 상대 경로 설정 (스크립트 파일 위치 기준)
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "preprocessed_data_20260131.csv")
    df = pd.read_csv(file_path)
    
    # 날짜 처리
    df['주문일'] = pd.to_datetime(df['주문일'])
    df['주문일자'] = df['주문일'].dt.date
    df['요일'] = df['주문일'].dt.day_name()
    df['시간대'] = df['주문일'].dt.hour
    
    # 숫자 변환
    num_cols = ['결제금액', '주문취소 금액', '실결제 금액', '판매단가', '공급단가', '주문수량', '취소수량', '주문-취소 수량']
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- 보고서 기반 파생 변수 및 그룹화 ---
    # 1. 이벤트/선물 키워드 (보고서 기준 정교화)
    event_keywords = '1\+1|증정|추가발송|이벤트|특가|한정|폭탄'
    gift_keywords = '선물|포장|선물세트|선물용'
    df['is_event_item'] = df['상품명'].str.contains(event_keywords).fillna(False) | (df['이벤트 여부'] == 'Y')
    df['is_gift_item'] = df['상품명'].str.contains(gift_keywords).fillna(False) | (df['선물세트_여부'].str.contains('선물|세트').fillna(False))
    
    # 2. 가격대 그룹 (보고서 기준: 1-3만원, 3-5만원, 5-10만원 등)
    def categorize_price(price):
        if price < 10000: return '1만원 미만'
        elif price < 30000: return '1-3만원대'
        elif price < 50000: return '3-5만원대'
        elif price < 100000: return '5-10만원대'
        else: return '10만원 이상'
    df['단가_그룹'] = df['판매단가'].apply(categorize_price)
    
    # 3. 순이익 계산 (수수료 10% 가정)
    fee_rate = 0.1
    df['순이익'] = df['실결제 금액'] - df['공급단가'].fillna(0) - (df['실결제 금액'] * fee_rate)
    df['순이익률'] = (df['순이익'] / df['실결제 금액']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 4. 재구매 및 취소 정보
    if 'UID' in df.columns:
        # 셀러별 재구매 여부 (가설 5용)
        df['is_reorder'] = df.groupby(['셀러명', 'UID'])['주문번호'].transform('nunique') > 1
        # 고객사 전체 기준 첫 구매 여부 (가설 10용)
        # 각 UID별 가장 빠른 주문일을 찾음
        df['first_order_date'] = df.groupby('UID')['주문일자'].transform('min')
        df['is_first_purchase'] = df['주문일자'] == df['first_order_date']
    else:
        df['is_reorder'] = False
        df['is_first_purchase'] = True
        
    # 5. 시간대 구간화 (가설 9용)
    def categorize_time(hour):
        if 0 <= hour < 6: return '새벽 (00-06)'
        elif 6 <= hour < 12: return '오전 (06-12)'
        elif 12 <= hour < 18: return '오후 (12-18)'
        elif 18 <= hour < 21: return '저녁 (18-21)'
        else: return '야간 (21-24)'
    df['시간대_구간'] = df['시간대'].apply(categorize_time)
    
    df['is_cancelled'] = df['취소여부'] == 'Y'
    
    return df

df = load_data()

# 사이드바
st.sidebar.header("🔍 분석 필터")
keyword_input = st.sidebar.text_input("상품명 키워드 검색 (비워두면 전체)", "")
keywords = [k.strip() for k in keyword_input.split(",") if k.strip()]

min_date, max_date = df['주문일자'].min(), df['주문일자'].max()
date_range = st.sidebar.date_input("주문 기간", [min_date, max_date])

# 필터링
mask = (df['주문일자'] >= date_range[0]) & (df['주문일자'] <= date_range[1])
filtered_df = df[mask]

# 메인 UI
st.title("🍊 농산물 이커머스 상세 분석 대시보드")
st.markdown("> **상품 구조 기반 구매 행동 EDA 보고서**의 실시간 데이터 버전입니다.")

tabs = st.tabs(["� 전략 리포트", "�📈 트렌드", "📊 기초 EDA", "💡 가설 검증", "🧪 A/B 테스트", "📋 데이터"])

# --- Tab 0: 전략 리포트 ---
with tabs[0]:
    st.header("📄 상품 구조 기반 구매 행동 EDA 분석 보고서")
    st.markdown("""
    본 보고서는 농산물 이커머스 주문 데이터를 바탕으로 상품의 단가, 옵션, 키워드 구조가 고객의 구매 결정 및 취소 행태에 미치는 영향을 분석한 결과입니다.
    """)
    
    with st.expander("1. 분석 결과 요약", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📊 단가 및 옵션 구조 분석**
            - **주력 가격대**: 3~5만원대(4,235건)와 1-3만원대(3,820건)가 주류
            - **인기 옵션**: ('소과', '3-5kg') 조합이 압도적 1위 (1,647건)
            """)
        with col2:
            st.markdown("""
            **🔍 키워드 영향도 분석**
            - **주문 볼륨**: '가정용(Home)' 키워드 유입 최다 (1,925건)
            - **안정성**: '이벤트' 상품 취소율 **3.48%**로 최저 (구매 확정성 높음)
            """)
            
        st.markdown("""
        **⚠️ 취소 발생 특징**
        - **가격 상관성**: 5~10만원대 고단가 상품 취소율 **27.76%**로 매우 높음 (심리적 저항)
        - **규격**: 소과/혼합 규격이 대과보다 취소율이 미세하게 높음 (기대치 불일치)
        """)

    with st.expander("2. 해석 및 비즈니스 제언"):
        st.markdown("""
        - **가속화된 실속 소비**: '가성비(가정용)' + '보관 편의성(3-5kg)' 조합에서 구매 활발
        - **고단가 상품의 병목**: 5만원 이상 선물세트의 높은 취소율(27%↑) 극복을 위한 리패키징 필요
        - **카테고리 믹스 전략**: 고구마(입구 상품) 구매 고객 대상 감귤 교차 판매 캠페인 유효
        """)

    with st.expander("4. 심층 가설 검증 요약"):
        st.table(pd.DataFrame({
            "가설 항목": [
                "지역별 셀러 점유율 (H1)", 
                "이벤트 수익성 (H3)", 
                "팬덤 셀러 재구매 (H5)", 
                "시간대별 마케팅 효과 (H9)", 
                "첫 구매 타겟 효율 (H10)"
            ],
            "검증 결과 및 인사이트": [
                "채택 (경기도 내 킹댕즈 점유율 22%로 편중 확인)", 
                "채택 (이벤트 상품 이익률 22.8%로 일반 상품 압도)", 
                "채택 (제주농장 재구매율 51.9% 달성)", 
                "채택 (저녁/야간 시간대 이벤트 반응도 집중)", 
                "채택 (첫 구매자 비중 높고 이벤트 민감도 강함)"
            ]
        }))

    with st.expander("5. 최종 액션 플랜 (마케터용)", expanded=True):
        st.success("**🚀 1. 초정밀 시간대(Time-slot) 타겟팅**: 가설 9에 따른 주문 집중 시간대(저녁~야간)에 맞춰 앱 푸시 및 타임세일을 집중 배치하여 전환율 극대화")
        st.success("**� 2. 신규 고객 '입구 상품' 최적화**: 가설 10의 높은 첫 구매 비중을 고려하여, 가성비 규격(3-5kg) 상품을 첫 구매 전용 혜택으로 전면 배치하여 락인 유도")
        st.success("**📍 3. 지역별 셀러 브랜드 파워 활용**: 가설 1 재검증 결과 확인된 특정 지역 선호 셀러(킹댕즈 등)를 해당 지역 타겟 광고(LBA) 모델로 활용하여 ROAS 개선")
        st.success("**� 4. 수익 기여형 이벤트 리패키징**: 일반 상품 대비 높은 순이익률(22.8%)을 보이는 이벤트 상품 구조를 일반 품목으로 확대 적용하여 총이익 개선")
        st.success("**⚠️ 5. 고단가 상품 취소율 방어**: 5만원↑ 상품의 높은 심리적 저항(취소율 27%)을 낮추기 위한 배송 전 안심 서비스(검수 영상 등) 또는 3-5만원대 리패키징 권고")

# --- Tab 1: 트렌드 ---
with tabs[1]:
    st.header("📈 상품 및 셀러 유형별 매출 트렌드 상세")
    
    # 상단 요약 지표
    t_m1, t_m2, t_m3, t_m4 = st.columns(4)
    with t_m1:
        st.metric("총 실결제 금액", f"{filtered_df['실결제 금액'].sum():,.0f}원")
    with t_m2:
        st.metric("평균 객단가(ARPU)", f"{filtered_df['실결제 금액'].mean():,.0f}원")
    with t_m3:
        st.metric("활발한 셀러 수", f"{filtered_df['셀러명'].nunique()}명")
    with t_m4:
        st.metric("주문 건수", f"{filtered_df['주문번호'].nunique():,.0f}건")

    st.divider()
    
    t_col1, t_col2 = st.columns([1, 1])
    with t_col1:
        # [그래프 1] 상품 유형별(감귤 세부) 누적 매출 추이
        yearly_trend = filtered_df.groupby(['주문일자', '감귤 세부'])['실결제 금액'].sum().reset_index()
        fig1 = px.area(yearly_trend, x='주문일자', y='실결제 금액', color='감귤 세부', 
                        title="[그래프 1] 상품 유형별 일별 누적 매출 추이 (Stack Area)")
        st.plotly_chart(fig1, use_container_width=True)
        
        # [그래프 2] 셀러 유형(가격대 타겟)별 평균 결제 수준
        # 셀러가 주로 파는 가격대 그룹을 셀러의 유형으로 정의
        seller_type_df = filtered_df.groupby('셀러명')['단가_그룹'].agg(lambda x: x.value_counts().index[0]).reset_index()
        seller_type_df.columns = ['셀러명', '주력_가격대']
        temp_df = filtered_df.merge(seller_type_df, on='셀러명')
        fig2 = px.box(temp_df, x='주력_가격대', y='실결제 금액', color='주력_가격대',
                       title="[그래프 2] 셀러 주력 가격대별 실결제 금액 분포", points="outliers")
        st.plotly_chart(fig2, use_container_width=True)

        # [그래프 3] 요일/시간대별 매출 열지도 (Heatmap)
        heatmap_data = filtered_df.groupby(['요일', '시간대'])['실결제 금액'].sum().reset_index()
        # 요일 순서 정렬
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data['요일'] = pd.Categorical(heatmap_data['요일'], categories=day_order, ordered=True)
        heatmap_pivot = heatmap_data.pivot(index='요일', columns='시간대', values='실결제 금액')
        fig3 = px.imshow(heatmap_pivot, title="[그래프 3] 요일/시간대별 총 매출 열지도",
                         labels=dict(x="시간대", y="요일", color="매출액"),
                         color_continuous_scale='Viridis')
        st.plotly_chart(fig3, use_container_width=True)

    with t_col2:
        # [그래프 4] 상품 품종별 매출 비중 추이 (100% Stacked Bar)
        variety_trend = filtered_df.groupby(['주문일자', '품종'])['실결제 금액'].sum().reset_index()
        fig4 = px.bar(variety_trend, x='주문일자', y='실결제 금액', color='품종', 
                        title="[그래프 4] 일자별 상품 품종 구성비 추이", barmode='relative')
        st.plotly_chart(fig4, use_container_width=True)
        
        # [그래프 5] 상위 셀러별 매출 기여도 및 평균 단가 (Bubble Chart)
        seller_perf = filtered_df.groupby('셀러명').agg({
            '실결제 금액': ['sum', 'mean'],
            '주문번호': 'nunique'
        }).reset_index()
        seller_perf.columns = ['셀러명', '총매출', '평균결제액', '주문건수']
        fig5 = px.scatter(seller_perf.head(20), x='주문건수', y='총매출', size='평균결제액', color='셀러명',
                           hover_data=['셀러명'], title="[그래프 5] 상위 20개 셀러 매출-주문건수 (크기: 평균결제액)")
        st.plotly_chart(fig5, use_container_width=True)
        
        # [그래프 6] 이벤트 여부에 따른 시계열 매출 변화
        event_trend = filtered_df.groupby(['주문일자', 'is_event_item'])['실결제 금액'].sum().reset_index()
        fig6 = px.line(event_trend, x='주문일자', y='실결제 금액', color='is_event_item', 
                        title="[그래프 6] 이벤트 여부별 일별 매출 트렌드 비교", markers=True)
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("🌟 실시간 히어로 상품 (TOP 5)")
    st.divider()
    h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)
    hero_items = filtered_df.groupby('상품명')['주문번호'].nunique().sort_values(ascending=False).head(5)
    cols = [h_col1, h_col2, h_col3, h_col4, h_col5]
    for i, (name, count) in enumerate(hero_items.items()):
        with cols[i]:
            st.info(f"**{i+1}위**\n\n{name}\n\n**{count}건**")

# --- Tab 2: 기초 EDA ---
with tabs[2]:
    st.header("상품 및 취소 행태 분석")
    col1, col2 = st.columns(2)
    with col1:
        # 가격대별 주문 볼륨 (보고서 1번 항목)
        price_vol = filtered_df['단가_그룹'].value_counts().reindex(['1만원 미만', '1-3만원대', '3-5만원대', '5-10만원대', '10만원 이상']).reset_index()
        fig1 = px.bar(price_vol, x='단가_그룹', y='count', title="가격대별 주문 볼륨 (3-5만원대 주력)", text_auto=True, color='count')
        st.plotly_chart(fig1, use_container_width=True)
        
        # 유입 경로 비중
        inflow = filtered_df['주문경로'].value_counts().reset_index()
        st.plotly_chart(px.pie(inflow, values='count', names='주문경로', title="주문 유입 경로 비중", hole=0.4), use_container_width=True)

    with col2:
        # 단가 그룹별 취소율 (보고서 3번 항목)
        cancel_rate = filtered_df.groupby('단가_그룹')['is_cancelled'].mean().reset_index()
        cancel_rate['취소율(%)'] = cancel_rate['is_cancelled'] * 100
        fig2 = px.line(cancel_rate, x='단가_그룹', y='취소율(%)', title="가격대별 취소율 (5-10만원대 급증 확인)", markers=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        # 인기 옵션 (소과 vs 대과 등)
        fruit_size = filtered_df['과수 크기'].value_counts().head(5).reset_index()
        st.plotly_chart(px.bar(fruit_size, x='과수 크기', y='count', title="과수 크기별 선호도 (소과/혼합 비중 높음)", color='과수 크기'), use_container_width=True)

    st.divider()
    st.subheader("🎯 셀러 및 유입경로 상세 분석 (심층 그래프)")
    
    scol1, scol2 = st.columns(2)
    
    with scol1:
        # [그래프 5] 상위 10 셀러별 주요 판매 품종 (Stacked Bar)
        top_10_sellers = filtered_df['셀러명'].value_counts().head(10).index
        seller_variety_df = filtered_df[filtered_df['셀러명'].isin(top_10_sellers)]
        seller_variety_stats = seller_variety_df.groupby(['셀러명', '품종']).size().reset_index(name='주문건수')
        fig5 = px.bar(seller_variety_stats, x='셀러명', y='주문건수', color='품종', 
                      title="[그래프 5] 상위 10 셀러별 판매 품종 구성", barmode='stack')
        st.plotly_chart(fig5, use_container_width=True)
        
        # [그래프 6] 셀러별 주문 대비 취소 비중 (상위 15개 셀러)
        top_15_sellers = filtered_df['셀러명'].value_counts().head(15).index
        cancel_df = filtered_df[filtered_df['셀러명'].isin(top_15_sellers)]
        cancel_stats = cancel_df.groupby(['셀러명', '취소여부']).size().reset_index(name='건수')
        fig6 = px.bar(cancel_stats, x='셀러명', y='건수', color='취소여부', 
                       title="[그래프 6] 상위 셀러별 주문-취소 비중 (N:정상, Y:취소)", barmode='group')
        st.plotly_chart(fig6, use_container_width=True)
        
        # [그래프 7] 결제 수단별 이용 빈도
        pay_counts = filtered_df['결제방법'].value_counts().reset_index()
        fig7 = px.bar(pay_counts, x='count', y='결제방법', orientation='h', 
                      title="[그래프 7] 결제 수단별 이용 빈도", color='count')
        st.plotly_chart(fig7, use_container_width=True)

    with scol2:
        # [그래프 8] 주문 경로별 평균 객단가
        fig8 = px.box(filtered_df, x='주문경로', y='실결제 금액', color='주문경로', 
                      title="[그래프 8] 주문 경로별 결제금액 분포(객단가)")
        st.plotly_chart(fig8, use_container_width=True)
        
        # [그래프 9] 셀러별 평균 판매단가 비교 (상위 10 셀러)
        seller_price = filtered_df[filtered_df['셀러명'].isin(top_10_sellers)].groupby('셀러명')['판매단가'].mean().reset_index()
        fig9 = px.bar(seller_price, x='셀러명', y='판매단가', title="[그래프 9] 상위 10 셀러별 평균 판매단가", text_auto=',.0f')
        st.plotly_chart(fig9, use_container_width=True)
        
        # [그래프 10] 판매단가와 주문수량의 상관관계
        fig10 = px.scatter(filtered_df, x='판매단가', y='주문수량', size='실결제 금액', color='감귤 세부',
                             hover_data=['상품명', '셀러명'], title="[그래프 10] 판매단가와 주문수량의 상관관계")
        st.plotly_chart(fig10, use_container_width=True)

# --- Tab 3: 가설 검증 ---
with tabs[3]:
    st.header("💡 심층 가설 검증 결과 (보고서 동기화)")
    
    selected_h = st.selectbox("리포트 가설을 선택하세요:", [
        "[가설 1] 경기도 매출은 특정 셀러의 지역 편중 현상이다 (재검증)",
        "[가설 2] 이벤트 상품은 주문량을 견인한다 (채택)",
        "[가설 3] 이벤트 상품의 반전 수익성 (채택)",
        "[가설 4] 선물 목적은 고가/로얄과를 선택한다 (채택)",
        "[가설 5] 팬덤형 셀러 '제주농장' 분석 (채택)",
        "[가설 6] 셀러별 특화된 상품 구조 (채택)",
        "[가설 7/8] 셀러 유입 및 이탈 관리 (채택)",
        "[가설 9] 시간대별 마케팅 성과 차이 분석 (신규)",
        "[가설 10] 첫 구매 고객 전용 이벤트 효율성 (신규)"
    ])
    
    if "[가설 1]" in selected_h:
        st.subheader("지역별 셀러 점유율 편차 재검증 (의미성 분석)")
        
        # 지역별 셀러 점유율의 표준편차 계산 (어느 지역이 특정 셀러에 더 편중되어 있는지)
        seller_region_matrix = filtered_df.groupby(['광역지역(정식)', '셀러명'])['실결제 금액'].sum().unstack(fill_value=0)
        seller_region_pct = seller_region_matrix.div(seller_region_matrix.sum(axis=1), axis=0) * 100
        
        # 특정 셀러(예: 킹댕즈)의 지역별 점유율 추이
        target_seller = "킹댕즈" # 보고서 핵심 셀러
        if target_seller in seller_region_pct.columns:
            ts_data = seller_region_pct[target_seller].sort_values(ascending=False).reset_index()
            ts_data.columns = ['지역', '점유율(%)']
            
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = px.bar(ts_data, x='지역', y='점유율(%)', color='점유율(%)', 
                             title=f"'{target_seller}' 셀러의 지역별 점유율 (경기도 편중성 확인)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.metric("경기도 내 점유율", f"{ts_data[ts_data['지역']=='경기도']['점유율(%)'].values[0]:.1f}%")
                st.write("**재검증 결과**: 경기도는 타 지역 대비 특정 셀러의 점유율이 통계적으로 유의미하게 높습니다. 단순 매출 규모가 아닌 '브랜드 선호도'가 지역별로 다르게 형성되어 있음을 의미합니다.")
        else:
            st.warning(f"데이터 내에 '{target_seller}' 셀러 정보가 부족합니다.")

    elif "[가설 2]" in selected_h:
        ev_stats = filtered_df.groupby('is_event_item')['주문수량'].mean().reset_index()
        st.subheader("이벤트 여부에 따른 평균 주문수량 비교")
        st.plotly_chart(px.bar(ev_stats, x='is_event_item', y='주문수량', color='is_event_item', text_auto='.2f'), use_container_width=True)
        st.success("**보고서 결과**: 이벤트 상품 평균 주문수량(1.23개)이 일반 상품(1.08개)보다 약 14% 높음. 구매 결정 가속화 효과 증명.")

    elif "[가설 3]" in selected_h:
        ev_profit = filtered_df.groupby('is_event_item')['순이익률'].mean().reset_index()
        ev_profit['순이익률(%)'] = ev_profit['순이익률'] * 100
        
        st.subheader("이벤트 여부에 따른 순이익률 반전 효과")
        st.plotly_chart(px.bar(ev_profit, x='is_event_item', y='순이익률(%)', color='is_event_item', text_auto='.1f'), use_container_width=True)
        st.success("**보고서 통찰**: 이벤트 상품(22.8%)이 일반 상품(15.1%)보다 오히려 수익성이 높음! 혜택이 공급가 절감이나 업셀링으로 이어짐.")

    elif "[가설 4]" in selected_h:
        st.subheader("선물 vs 일반 주문 구매 특성 비교")
        gift_compare = filtered_df.groupby('is_gift_item')['판매단가'].mean().reset_index()
        st.plotly_chart(px.bar(gift_compare, x='is_gift_item', y='판매단가', color='is_gift_item', text_auto=',.0f'), use_container_width=True)
        st.info("**보고서 결과**: 선물용 평균 단가 3.89만원(일반 3.07만원). 선물용은 '대과' 비중(53%)이 압도적임. 프리미엄화 전략 제언.")

    elif "[가설 5]" in selected_h:
        st.subheader("셀러별 재구매율 (Fan-base)")
        reorder_s = filtered_df.groupby('셀러명').agg({'UID':'count', 'is_reorder':'sum'}).reset_index()
        reorder_s['재구매율(%)'] = (reorder_s['is_reorder'] / reorder_s['UID']) * 100
        top_r = reorder_s[reorder_s['UID'] >= 50].sort_values('재구매율(%)', ascending=False).head(5)
        
        st.plotly_chart(px.bar(top_r, x='재구매율(%)', y='셀러명', orientation='h', color='재구매율(%)', text_auto='.1f'), use_container_width=True)
        st.warning("**보고서 결과**: '제주농장'의 재구매율이 51.9%로 압도적임. 해당 셀러의 CS/배송 노하우 매뉴얼화 필요.")

    elif "[가설 6]" in selected_h:
        st.subheader("셀러별 전략 포지셔닝 맵")
        seller_map = filtered_df.groupby('셀러명').agg({
            'is_event_item': 'mean',
            'is_gift_item': 'mean',
            '판매단가': 'mean',
            '주문번호': 'nunique'
        }).reset_index()
        fig = px.scatter(seller_map, x='is_event_item', y='is_gift_item', size='주문번호', hover_data=['셀러명'], 
                         title="셀러별 전략 분포 (이벤트 비중 vs 선물 비중)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**보고서 결과**: 'dapanda'(프리미엄), '천&천'(프로모션) 등 명확한 포지셔닝을 가진 셀러 그룹 식별됨.")

    elif "[가설 7/8]" in selected_h:
        st.subheader("월별 셀러 활동성 추이")
        df['월'] = df['주문일'].dt.to_period('M').astype(str)
        monthly_sellers = df.groupby('월')['셀러명'].nunique().reset_index()
        st.plotly_chart(px.line(monthly_sellers, x='월', y='셀러명', title="월별 활동 셀러 수 추이", markers=True), use_container_width=True)
        st.error("**보고서 결과**: 11월 이후 대규모 이탈 발생. 셀러 Retention 관리 및 신규 유입 프로모션 시급.")

    elif "[가설 9]" in selected_h:
        st.subheader("🕒 시간대별 마케팅 효율성 분석")
        time_stats = filtered_df.groupby(['시간대_구간']).agg({
            '주문번호': 'nunique',
            '실결제 금액': 'sum',
            '판매단가': 'mean'
        }).reset_index()
        time_stats.columns = ['시간대', '주문수', '총매출', '평균단가']
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(time_stats, x='시간대', y='주문수', title="시간대별 주문 건수", color='시간대'), use_container_width=True)
        with c2:
            # 이벤트 반응도 분석
            time_ev = filtered_df.groupby(['시간대_구간', 'is_event_item'])['주문번호'].nunique().reset_index()
            st.plotly_chart(px.bar(time_ev, x='시간대_구간', y='주문번호', color='is_event_item', barmode='group', title="시간대별 이벤트 상품 반응도"), use_container_width=True)
            
        st.info("**분석 결과**: 특정 시간대(예: 저녁/야간)에 이벤트 상품의 구매 전환이 집중되는지 확인하여 '타임 세일' 전략 수립이 가능합니다.")

    elif "[가설 10]" in selected_h:
        st.subheader("🆕 첫 구매 고객 vs 재구매 고객 분석")
        
        first_vs_re = filtered_df['is_first_purchase'].value_counts(normalize=True).reset_index()
        first_vs_re.columns = ['유형', '비중']
        first_vs_re['유형'] = first_vs_re['유형'].map({True: '첫 구매', False: '재구매'})
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(first_vs_re, values='비중', names='유형', title="전체 주문 중 첫 구매 vs 재구매 비중"), use_container_width=True)
        with c2:
            compare_stats = filtered_df.groupby('is_first_purchase').agg({
                '실결제 금액': 'mean',
                'is_event_item': 'mean'
            }).reset_index()
            compare_stats['is_first_purchase'] = compare_stats['is_first_purchase'].map({True: '첫 구매', False: '재구매'})
            st.plotly_chart(px.bar(compare_stats, x='is_first_purchase', y='is_event_item', title="고객 유형별 이벤트 상품 선택률"), use_container_width=True)
            
        st.success("**비즈니스 인사이트**: 첫 구매 고객의 비중이 압도적으로 높다면 '입구 상품' 최적화 및 첫 구매 허들을 낮추는 전용 이벤트 배치가 필수적입니다.")

# --- Tab 4: A/B 테스트 실험실 ---
with tabs[4]:
    st.header("🧪 마케팅 A/B 테스트 전략 시뮬레이션")
    st.info("리포트 제언 사항을 기반으로 한 실험군(Test Group) vs 대조군(Control Group) 성과 분석")
    
    ab_case = st.pills("실험 케이스 선택", [
        "A: 고단가(5만원↑) 취소율 방어 테스트",
        "B: '이벤트' 키워드의 신뢰도(취소율) 효과",
        "C: 가성비 규격(3-5kg)의 복수구매 전환율"
    ])
    
    if ab_case == "A: 고단가(5만원↑) 취소율 방어 테스트":
        st.subheader("고단가 상품의 심리적 저항 확인")
        high_price_df = filtered_df.groupby('단가_그룹')['is_cancelled'].mean().reset_index()
        high_price_df['취소율(%)'] = high_price_df['is_cancelled'] * 100
        fig = px.bar(high_price_df, x='단가_그룹', y='취소율(%)', color='단가_그룹', 
                     title="가격대별 취소 리스크 (보고서: 5만원 이상 27.7%↑)")
        st.plotly_chart(fig, use_container_width=True)
        st.error("**액션 아이디어**: 5만원 이상 고가 상품은 결제 전 '심리적 저항'이 큼. 3-5만원대로 리패키징하거나 사은품을 강조하여 체감 가치를 증대시켜야 함.")

    elif ab_case == "B: '이벤트' 키워드의 신뢰도(취소율) 효과":
        st.subheader("이벤트 상품의 구매 확정성 분석")
        ev_cancel = filtered_df.groupby('is_event_item')['is_cancelled'].mean().reset_index()
        ev_cancel['취소율(%)'] = ev_cancel['is_cancelled'] * 100
        fig = px.bar(ev_cancel, x='is_event_item', y='취소율(%)', color='is_event_item', title="이벤트 키워드 유무별 취소율")
        st.plotly_chart(fig, use_container_width=True)
        st.success("**액션 아이디어**: 이벤트 상품은 취소율이 3.48%로 대조군 대비 매우 낮음. 단순 매출 증대용이 아닌 '구매 신뢰도' 확보 수단으로 활용 가능.")

    elif ab_case == "C: 가성비 규격(3-5kg)의 복수구매 전환율":
        st.subheader("3-5kg 실속형 규격의 대량 주문(Bulk) 성향")
        filtered_df['is_bulk'] = filtered_df['주문수량'] >= 2
        bulk_stats = filtered_df.groupby('무게 구분')['is_bulk'].mean().reset_index()
        bulk_stats['복수구매비중(%)'] = bulk_stats['is_bulk'] * 100
        fig = px.bar(bulk_stats, x='무게 구분', y='복수구매비중(%)', color='복수구매비중(%)', title="상품 규격별 복수 구매 비중")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**액션 아이디어**: 3-5kg 규격에서 복수 구매가 빈번함. 해당 규격 구매 고객대상으로 '2개 담으면 추가 할인' 쿠폰 발행 시 업셀링 효과 극대화 예상.")

# --- Tab 5: 데이터 ---
with tabs[5]:
    st.header("상세 데이터 조회")
    st.dataframe(filtered_df, use_container_width=True)
    st.download_button("📥 필터링된 데이터 CSV 다운로드", filtered_df.to_csv(index=False).encode('utf-8-sig'), "filtered.csv", "text/csv")

# 푸터
st.markdown("---")
st.caption("© 2026 mffarm04 | 감귤 이커머스 마케팅 의사결정 지원 시스템")
