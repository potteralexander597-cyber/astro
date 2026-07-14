import numpy as np
import pandas as pd
import plotly.graph_objects as gr
import streamlit as st
from scipy.integrate import solve_bvp, solve_ivp

# ==============================================================================
# CONFIGURATION & PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="대기·해양 동역학 시뮬레이터",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌌 대기·해양 역학: 경계층 물리 및 3차원 에크만 수송 시뮬레이터")
st.markdown(
    """
    본 시뮬레이터는 고등학교 지구과학 II의 '힘의 평형' 개념을 넘어, 
    **대기 경계층(ABL)의 연립 미분방정식 수치 해석**, **에크만 펌핑의 와도 수치 미분(FDM)**, 
    그리고 **RK4 기반의 시간 의존성 바람 장 변화**를 실시간으로 모델링하는 학부 초년생 수준의 고등 동역학 앱입니다.
    """
)
st.sidebar.header("🎛️ 전역 및 물리 매개변수 설정")

# 전역 변수 설정 (위도 및 코리올리 파라미터)
latitude = st.sidebar.slider("위도 (Latitude, °N)", 10.0, 80.0, 35.0, step=0.5)
omega = 7.2921e-5  # 지구 자전 각속도 (rad/s)
f = 2 * omega * np.sin(np.radians(latitude))
rho_a = 1.225  # 대기 밀도 (kg/m^3)
rho_w = 1025.0  # 해수 밀도 (kg/m^3)

st.sidebar.markdown(
    f"**Coriolis Parameter ($f$):** `{f:.4e} s⁻¹`"
)

# 탭 구성을 통해 모듈 분할
tab1, tab2, tab3 = st.tabs(
    [
        "💡 모듈 A: 에크만 나선 (Ekman Spiral)",
        "💡 모듈 B: 에크만 펌핑 (Ekman Pumping)",
        "💡 모듈 C: 시간 의존성 바람 장 (Time-dependent Adjustment)",
    ]
)

# ==============================================================================
# MODULE A: EKMAN SPIRAL (BVP SOLVER)
# ==============================================================================
with tab1:
    st.header("모듈 A: 에크만 나선(Ekman Spiral)의 연직 연속적 수치 해법")
    st.markdown(
        "연직 에디 점성 계수($K_m$)와 거칠기 길이($z_0$)를 반영하여 연직 고도에 따른 네비어-스토크스 간소화 방정식의 **경계값 문제(BVP)**를 수치 해석합니다."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 모듈 A 매개변수")
        system_type = st.radio("시뮬레이션 대상 선택", ["대기 (Atmosphere)", "해양 (Ocean)"])
        K_m = st.slider(
            "연직 에디 점성 계수 ($K_m$, m²/s)", 0.1, 50.0, 5.0, step=0.1
        )

        if system_type == "대기 (Atmosphere)":
            ug_val = st.slider(
                "지균풍 동서 성분 ($u_g$, m/s)", -30.0, 30.0, 15.0, step=1.0
            )
            vg_val = st.slider(
                "지균풍 남북 성분 ($v_g$, m/s)", -30.0, 30.0, 0.0, step=1.0
            )
            z_max = st.slider(
                "최대 고도 (Boundary Layer Height, m)", 500, 3000, 1500, step=100
            )
            z_0 = st.number_input("지표 거칠기 길이 ($z_0$, m)", value=0.1, format="%f")
        else:
            ug_val = st.slider(
                "지균 해류 동서 성분 ($u_g$, m/s)", -2.0, 2.0, 0.5, step=0.1
            )
            vg_val = st.slider(
                "지균 해류 남북 성분 ($v_g$, m/s)", -2.0, 2.0, 0.0, step=0.1
            )
            z_max = st.slider("최대 수심 (Ekman Depth, m)", 20, 200, 100, step=5)
            z_0 = 0.0

    # BVP 수치 해석 수행
    # d4u/dz4 형태로 변환하거나 1차 연립 미분방정식계로 변환
    # y = [u, du/dz, v, dv/dz]
    def ekman_bvp_sys(z, y):
        u, dudz, v, dvdz = y
        # f(v - vg) + Km * d2u/dz2 = 0  =>  d2u/dz2 = -f(v - vg)/Km
        # -f(u - ug) + Km * d2v/dz2 = 0 =>  d2v/dz2 = f(u - ug)/Km
        d2u_dz2 = -f * (v - vg_val) / K_m
        d2v_dz2 = f * (u - ug_val) / K_m
        return [dudz, d2u_dz2, dvdz, d2v_dz2]

    def ekman_bc(ya, yb):
        if system_type == "대기 (Atmosphere)":
            # 하부 경계: z=z_0에서 u=0, v=0 / 상부 경계: z=z_max에서 u=ug, v=vg
            return [ya[0], ya[2], yb[0] - ug_val, yb[2] - vg_val]
        else:
            # 해양 하부 경계(심층): u=0, v=0 / 상부 경계(표면): 풍응력 대신 표면 유속을 ug, vg로 가정하거나 변형 유도
            return [ya[0] - ug_val, ya[2] - vg_val, yb[0], yb[2]]

    z_steps = np.linspace(z_0, z_max, 200)
    y_guess = np.zeros((4, z_steps.size))

    res_bvp = solve_bvp(ekman_bvp_sys, ekman_bc, z_steps, y_guess)

    with col2:
        if res_bvp.success:
            z_sol = res_bvp.x
            u_sol = res_bvp.y[0]
            v_sol = res_bvp.y[2]

            # 3D Plotly 시각화
            fig_a = gr.Figure()
            fig_a.add_trace(
                gr.Scatter3d(
                    x=u_sol,
                    y=v_sol,
                    z=z_sol,
                    mode="lines+markers",
                    marker=dict(
                        size=3,
                        color=z_sol,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="고도/수심 (m)"),
                    ),
                    line=dict(color="darkblue", width=4),
                    name="Ekman Spiral",
                )
            )

            # 지균풍/지균해류 기준선 표시
            fig_a.add_trace(
                gr.Scatter3d(
                    x=[ug_val, ug_val],
                    y=[vg_val, vg_val],
                    z=[0, z_max],
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name="Geostrophic Flow",
                )
            )

            fig_a.update_layout(
                scene=dict(
                    xaxis_title="U 성분 (m/s)",
                    yaxis_title="V 성분 (m/s)",
                    zaxis_title="연직 위치 (m)",
                ),
                title="3차원 에크만 나선 구조 공간 벡터 구조물",
                margin=dict(l=0, r=0, b=0, t=40),
                height=600,
            )
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.error("BVP Solver가 수렴하지 못했습니다. 매개변수를 조정해 주세요.")

# ==============================================================================
# MODULE B: EKMAN PUMPING (FDM VORTICITY SOLVER)
# ==============================================================================
with tab2:
    st.header("모듈 B: 에크만 펌핑(Ekman Pumping)과 유체 연직 운동 시뮬레이션")
    st.markdown(
        "2차원 격자 공간에서 임의의 기압 배치 및 풍속장의 와도($\\nabla \\times \\vec{V}$)를 **유한차분법(FDM)**으로 연산하여 연직 용승/침강 속도($w_e$)를 유도합니다."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 모듈 B 매개변수")
        system_mode = st.selectbox(
            "기압계 메커니즘 유형 고르기", ["저기압성 소용돌이 (Cyclone)", "고기압성 소용돌이 (Anticyclone)"]
        )
        vortex_scale = st.slider("시스템 반경 크기 (Scale, km)", 100, 1000, 500, step=50)
        max_wind = st.slider("최대 풍속 속력 (m/s)", 5, 60, 30, step=5)

    # 2D 격자 설정
    L = 1500 * 1000  # 영역 크기 (1500 km)
    N = 50  # 격자 수
    x = np.linspace(-L / 2, L / 2, N)
    y = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2) + 1e-5

    # 가우시안 형태의 소용돌이 풍속장 생성
    R_scale = vortex_scale * 1000
    if system_mode == "저기압성 소용돌이 (Cyclone)":
        # 반시계 방향 수렴성 바람 모사 (대기 지상풍 기준 수렴각 적용)
        theta = np.arctan2(Y, X)
        v_theta = max_wind * (R / R_scale) * np.exp(-((R / R_scale) ** 2))
        # 마찰각 30도 가정하여 안쪽으로 불어 들어가게 유도
        alpha = np.radians(30)
        U_field = -v_theta * np.sin(theta + alpha)
        V_field = v_theta * np.cos(theta + alpha)
    else:
        # 시계 방향 발산성 바람 모사
        theta = np.arctan2(Y, X)
        v_theta = -max_wind * (R / R_scale) * np.exp(-((R / R_scale) ** 2))
        alpha = np.radians(30)
        U_field = -v_theta * np.sin(theta - alpha)
        V_field = v_theta * np.cos(theta - alpha)

    # 중심 기압 경도에 따른 대기 표면 응력(Tau) 계산 (Cd 대입식)
    C_d = 1.5e-3
    tau_x = rho_a * C_d * np.sqrt(U_field**2 + V_field**2) * U_field
    tau_y = rho_a * C_d * np.sqrt(U_field**2 + V_field**2) * V_field

    # 유한차분법 (FDM Central Difference) 기반 Curl 계산
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    dtauy_dx = np.gradient(tau_y, axis=1) / dx
    dtaux_dy = np.gradient(tau_x, axis=0) / dy
    curl_tau = dtauy_dx - dtaux_dy

    # 에크만 펌핑 연직 속도 계산 (w_e = curl(tau) / (rho * f))
    w_e = curl_tau / (rho_a * f) * 100  # 시각화를 위해 cm/s 단위 변환

    with col2:
        # Plotly Contour & Streamline 시각화
        fig_b = gr.Figure()

        # 배경 2D 등고선맵 (연직 속도 분포)
        fig_b.add_trace(
            gr.Contour(
                z=w_e,
                x=x / 1000,
                y=y / 1000,
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="연직 속도 w_e (cm/s)"),
            )
        )

       # 바람 벡터장 화살표 추가 (Streamtube 대신 3D 공간 벡터 표현에 적합한 Cone 사용)
        skip = 3
        fig_b.add_trace(
            gr.Cone(
                x=X[::skip, ::skip].flatten() / 1000,
                y=Y[::skip, ::skip].flatten() / 1000,
                z=np.zeros_like(X[::skip, ::skip].flatten()), # 평면 z=0 위에 배치
                u=U_field[::skip, ::skip].flatten(),
                v=V_field[::skip, ::skip].flatten(),
                w=w_e[::skip, ::skip].flatten() / 100,       # 연직 속도 반영
                sizemode="scaled",
                sizeref=2.0,
                colorscale="Portland",
                showscale=False,
                name="Wind Vector"
            )
        )

        fig_b.update_layout(
            title="에크만 수송에 의한 하층 발산·수렴 및 연직 운동(w_e) 분포",
            xaxis_title="X 거리 (km)",
            yaxis_title="Y 거리 (km)",
            height=600,
        )
        st.plotly_chart(fig_b, use_container_width=True)

# ==============================================================================
# MODULE C: TIME-DEPENDENT WIND ADJUSTMENT (ODE SOLVER)
# ==============================================================================
with tab3:
    st.header("모듈 C: 시간에 따른 바람 장의 변화 및 관성 진동 추적")
    st.markdown(
        "정지 상태의 공기 덩어리에 연속 기압 경도력이 가해질 때 코리올리 힘과 지표 마찰력의 상호작용으로 **지균풍/지상풍 평형으로 조절되는 시계열**을 RK4 수치기법으로 추적합니다."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 모듈 C 매개변수")
        p_grad_x = st.slider(
            "X방향 기압경도력 ($\\frac{1}{\\rho}\\frac{\\partial P}{\\partial x}$, m/s²)",
            1e-4,
            20e-4,
            5e-4,
            format="%.1e",
        )
        r_friction = st.slider(
            "선형 마찰 계수 ($r$, s⁻¹)", 0.0, 2e-4, 5e-5, format="%.1e"
        )
        t_max_hours = st.slider(
            "수치 적분 가동 시간 (Simulation Time, hours)", 12, 120, 72, step=6
        )

    # ODE 지배 방정식 정의
    def wind_adjustment_system(t, state):
        u, v = state
        # du/dt = f*v - (1/rho)*dP/dx - r*u  (단, dP/dy=0 가정)
        du_dt = f * v + p_grad_x - r_friction * u
        dv_dt = -f * u - r_friction * v
        return [du_dt, dv_dt]

    # IVP 수치 적분 수행
    t_span = (0, t_max_hours * 3600)
    t_eval = np.linspace(0, t_max_hours * 3600, 1000)
    initial_condition = [0.0, 0.0]  # 정지 상태에서 출발

    sol_ode = solve_ivp(
        wind_adjustment_system,
        t_span,
        initial_condition,
        t_eval=t_eval,
        method="RK45",
    )

    with col2:
        if sol_ode.success:
            u_t = sol_ode.y[0]
            v_t = sol_ode.y[1]
            time_hours = sol_ode.t / 3600

            # 최종 평형 상태(이론값 계산)
            # 0 = f*v_eq + p_grad - r*u_eq
            # 0 = -f*u_eq - r*v_eq => v_eq = -f/r * u_eq
            denom = f**2 + r_friction**2
            u_eq = (p_grad_x * r_friction) / denom if r_friction > 0 else 0
            v_eq = -(p_grad_x * f) / denom

            # Plotly 2D 호도그래프(Hodograph) 시각화
            fig_c = gr.Figure()

            # 공기 덩어리 바람 벡터 궤적
            fig_c.add_trace(
                gr.Scatter(
                    x=u_t,
                    y=v_t,
                    mode="lines+markers",
                    marker=dict(
                        size=3,
                        color=time_hours,
                        colorscale="Jet",
                        colorbar=dict(title="경과 시간 (Hours)"),
                    ),
                    line=dict(color="black", width=2),
                    name="시간적 바람 궤적 (Trajectory)",
                )
            )

            # 수렴 지점 마커 (최종 평형 상태)
            fig_c.add_trace(
                gr.Scatter(
                    x=[u_eq],
                    y=[v_eq],
                    mode="markers",
                    marker=dict(color="red", size=12, symbol="star"),
                    name="최종 역학적 평형점 (Steady State)",
                )
            )

            fig_c.update_layout(
                title="풍속 호도그래프(Hodograph) 공간 내 관성 진동 수렴 상도",
                xaxis_title="U 풍속 (m/s)",
                yaxis_title="V 풍속 (m/s)",
                height=500,
            )
            st.plotly_chart(fig_c, use_container_width=True)

            # 시계열 데이터 요약 정보 출력
            st.info(
                f"🎯 **분석 결과 요약:** 설정한 물리계 하에서 초기 정지 상태의 공기는 전향력과 마찰력의 진동 상호작용을 거쳐 "
                f"최종 평형 풍속 **U: {u_eq:.2f} m/s, V: {v_eq:.2f} m/s**로 점근적 수렴(관성 진동 감쇄)합니다."
            )
