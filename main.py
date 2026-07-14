import numpy as np
import pandas as pd
import plotly.graph_objects as gr
import streamlit as st
from scipy.integrate import solve_bvp, solve_ivp

# ==============================================================================
# CONFIGURATION & PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="대기·해양 고등 동역학 시뮬레이터",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌌 대기·해양 역학: 경계층 물리, 에크만 펌핑 및 비선형 경도풍 적응 시뮬레이터")
st.markdown(
    """
    본 시뮬레이터는 고등학교 지구과학 II 및 대학 학부 대기동역학/물리해양학 과정을 아우르는 고등 수치 해석 앱입니다. 
    단순한 이상적 평형 계산을 넘어, **실제 경계층 마찰 효과, 유한차분법(FDM) 기반 와도 해석, 비선형 경도풍 점근 수렴 과정**을 시각적으로 규명합니다.
    """
)

# ------------------------------------------------------------------------------
# SIDEBAR: GLOBAL PARAMETERS
# ------------------------------------------------------------------------------
st.sidebar.header("🎛️ 전역 물리 매개변수 설정")
latitude = st.sidebar.slider("위도 (Latitude, °N)", 10.0, 80.0, 35.0, step=0.5)
omega = 7.2921e-5  # 지구 자전 각속도 (rad/s)
f = 2 * omega * np.sin(np.radians(latitude))
rho_a = 1.225      # 대기 밀도 (kg/m^3)
rho_w = 1025.0     # 해수 밀도 (kg/m^3)

st.sidebar.markdown(
    f"""
    ---
    **코리올리 매개변수 ($f$):**  
    `{f:.5e} s⁻¹`  
    *(위도에 따른 지구 자전 효과의 연직 성분 크기)*
    """
)

# 탭 구성
tab1, tab2, tab3 = st.tabs(
    [
        "💡 모듈 A: 에크만 나선 (Ekman Spiral)",
        "💡 모듈 B: 에크만 펌핑 (Ekman Pumping)",
        "💡 모듈 C: 경도풍 비선형 적응 (Gradient Wind Adjustment)",
    ]
)

# ==============================================================================
# MODULE A: EKMAN SPIRAL (BVP SOLVER) - 거칠기 효과 극대화
# ==============================================================================
with tab1:
    st.header("모듈 A: 에크만 나선(Ekman Spiral)과 지표 거칠기 길이($z_0$)의 연직 수치 해법")
    st.markdown(
        """
        ### 📌 개요 및 지배 방정식
        대기 경계층(ABL) 내에서 풍속은 지표 마찰력, 전향력, 기압경도력의 3력 균형에 의해 고도에 따라 크기와 방향이 변하며 나선형 궤적을 그리게 됩니다.
        
        $$f(v - v_g) + K_m \\frac{\\partial^2 u}{\\partial z^2} = 0$$
        $$-f(u - u_g) + K_m \\frac{\\partial^2 v}{\\partial z^2} = 0$$
        
        여기서 $K_m$은 **연직 에디 점성 계수**, $z_0$는 **지표 거칠기 길이(Roughness length)**입니다. 
        본 모듈은 거칠기 길이($z_0$)가 미치는 하부 경계 효과를 수치적으로 명확히 규명하기 위해, 고도 $z$의 범위를 $z_0$부터 시작하는 연립 이계 상미분 방정식 경계값 문제(BVP)로 공식화하여 정밀 해석합니다.
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 모듈 A 제어 변수")
        system_type = st.radio("물리계 선택", ["대기 경계층 (Atmosphere)", "해양 혼합층 (Ocean)"])
        
        K_m = st.slider(
            "연직 에디 점성 계수 ($K_m$, m²/s)", 0.1, 30.0, 5.0, step=0.1,
            help="공기의 난류 점성을 나타내며, 이 값이 클수록 마찰 영향이 상층까지 강하게 전달됩니다."
        )
        
        # 거칠기에 따른 극적인 효과 변화를 유도하기 위한 입력부
        z_0_preset = st.selectbox(
            "지표 거칠기 유형 ($z_0$ 사전 설정)",
            ["매끄러운 해수면 (0.0001m)", "개활지 및 초원 (0.01m)", "낮은 수목 지대 (0.1m)", "숲 및 도시 지역 (1.0m)", "고층 빌딩 밀집 지역 (3.0m)"]
        )
        
        # 사전 설정에 따른 z_0 매핑
        if "해수면" in z_0_preset:
            z_0_val = 0.0001
        elif "개활지" in z_0_preset:
            z_0_val = 0.01
        elif "수목" in z_0_preset:
            z_0_val = 0.1
        elif "숲" in z_0_preset:
            z_0_val = 1.0
        else:
            z_0_val = 3.0
            
        ug_val = st.slider("자유 대기 지균풍 ($u_g$, m/s)", -25.0, 25.0, 15.0, step=1.0)
        vg_val = st.slider("자유 대기 지균풍 ($v_g$, m/s)", -25.0, 25.0, 0.0, step=1.0)
        z_max = st.slider("관측 최대 고도 ($z_{max}$, m)", 200, 2000, 1000, step=100)

    # BVP Solver 구현: y = [u, du/dz, v, dv/dz]
    def ekman_bvp_sys(z, y):
        u, dudz, v, dvdz = y
        d2u_dz2 = -f * (v - vg_val) / K_m
        d2v_dz2 = f * (u - ug_val) / K_m
        return [dudz, d2u_dz2, dvdz, d2v_dz2]

    def ekman_bc(ya, yb):
        if system_type == "대기 경계층 (Atmosphere)":
            # 하부 경계: z = z_0에서 u = 0, v = 0 (점착 조건)
            # 상부 경계: z = z_max에서 지균풍 수렴 (u = ug, v = vg)
            return [ya[0], ya[2], yb[0] - ug_val, yb[2] - vg_val]
        else:
            # 해양: 표면에서 바람 응력 효과가 지균해류를 유도하는 구조적 거동 모사
            return [ya[0] - ug_val, ya[2] - vg_val, yb[0], yb[2]]

    # 고도 그리드 구성: 거칠기 길이 z_0에서 시작하여 z_max까지 조밀하게 분할
    z_steps = np.logspace(np.log10(z_0_val), np.log10(z_max), 300)
    y_guess = np.zeros((4, z_steps.size))

    res_bvp = solve_bvp(ekman_bvp_sys, ekman_bc, z_steps, y_guess)

    with col2:
        if res_bvp.success:
            z_sol = res_bvp.x
            u_sol = res_bvp.y[0]
            v_sol = res_bvp.y[2]

            # 바람 벡터 시각화 (3D Line)
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
                        colorscale="Thermal",
                        showscale=True,
                        colorbar=dict(title="고도 $z$ (m)", x=1.15),
                    ),
                    line=dict(color="darkblue", width=5),
                    name="Ekman Spiral",
                )
            )

            # 지균풍 벡터 점선 표시
            fig_a.add_trace(
                gr.Scatter3d(
                    x=[ug_val, ug_val],
                    y=[vg_val, vg_val],
                    z=[z_0_val, z_max],
                    mode="lines",
                    line=dict(color="red", width=3, dash="dash"),
                    name="Geostrophic Limit (지균풍)",
                )
            )

            # 고도별 대표 화살표 벡터 (Cone) 추가
            cone_skip = 15
            fig_a.add_trace(
                gr.Cone(
                    x=u_sol[::cone_skip],
                    y=v_sol[::cone_skip],
                    z=z_sol[::cone_skip],
                    u=u_sol[::cone_skip],
                    v=v_sol[::cone_skip],
                    w=np.zeros_like(z_sol[::cone_skip]),
                    sizemode="scaled",
                    sizeref=1.5,
                    colorscale="Blues",
                    showscale=False,
                    name="고도별 풍향/풍속 벡터"
                )
            )

            fig_a.update_layout(
                scene=dict(
                    xaxis_title="U 풍속 성분 (m/s)",
                    yaxis_title="V 풍속 성분 (m/s)",
                    zaxis_title="고도 z (m)",
                    zaxis=dict(range=[0, z_max]),
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=1.2)
                ),
                title=f"3차원 에크만 나선 시각화 (지표 거칠기 $z_0$ = {z_0_val} m)",
                margin=dict(l=0, r=0, b=0, t=40),
                height=600,
            )
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.error("BVP 수치 해석 솔버가 수렴에 실패했습니다. 입력 매개변수를 점검하세요.")

    st.markdown(
        f"""
        ---
        ### 📝 모듈 A 동역학적 핵심 메커니즘 해설 (가독성 요약)
        *   **지표 거칠기 길이($z_0$)의 파급 효과**: 
            *   **$z_0$가 매우 작을 때(예: 매끄러운 해수면)**: 마찰 효과가 지면 극근처에만 국한되어, 고도가 조금만 상승해도 바람이 지균풍 균형($u_g$, $v_g$)으로 빠르게 복귀하며 에크만 나선의 꼬임 반경이 매우 얇아집니다.
            *   **$z_0$가 클 때(예: 대도시 빌딩숲)**: 지상 거칠기가 상층 흐름에 미치는 저항 마찰력의 크기가 커지며, **자유 대기 지균풍에 도달하기 위한 마찰 경계층의 두께가 비약적으로 증가**합니다. 이에 따라 나선의 크기가 거대해지며 회전 각도가 지면 근처에서 아주 크게 꺾입니다.
        *   **에크만 전향각(Turn Angle)**: 이론적으로 지면($z \\to z_0$)에 가까워질수록 마찰력의 영향으로 바람은 등압선(지균풍 방향)에 대해 **북반구 기준 좌측 약 45°**까지 꺾여 불게 되며, 고도가 올라감에 따라 마찰이 줄어들어 점차 시계 방향으로 회전하며 지균풍에 수렴합니다.
        """
    )

# ==============================================================================
# MODULE B: EKMAN PUMPING (FDM VORTICITY SOLVER) - 매개변수 다양화
# ==============================================================================
with tab2:
    st.header("모듈 B: 에크만 펌핑(Ekman Pumping)과 복합 기압 배치 시뮬레이터")
    st.markdown(
        """
        ### 📌 개요 및 지배 방정식
        경계층 내 마찰에 의해 지상풍이 저기압 중심으로 수렴하거나 고기압 중심에서 발산할 때, 질량 보존 법칙(연속 방정식)에 의해 연직 운동이 강제됩니다. 
        이 현상을 **에크만 펌핑(Ekman Pumping, 용승 및 침강)**이라고 합니다.
        
        $$w_e = \\frac{1}{\\rho_a f} \\nabla \\times \\vec{\\tau} = \\frac{1}{\\rho_a f} \\left( \\frac{\\partial \\tau_y}{\\partial x} - \\frac{\\partial \\tau_x}{\\partial y} \right)$$
        
        $$\\vec{\\tau} = \\rho_a C_d |\\vec{V}| \\vec{V}$$
        
        여기서 $C_d$는 **항력 계수(Drag Coefficient)**, $\\vec{\\tau}$는 **지표 바람 응력 벡터**입니다. 
        본 모듈에서는 마찰 항력 계수와 기압 중심의 이심률(타원도), 다중 기압계 병합 등 현실적인 경계조건들을 가미하여 와도를 수치 미분(FDM)합니다.
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 모듈 B 기압계 제어 변수")
        pressure_system = st.selectbox(
            "기압 배치 시스템 선택",
            ["원형 저기압 (Typhoon)", "원형 고기압 (Anticyclone)", "타원형 기압골 (Elliptical Trough)"]
        )
        
        max_wind_speed = st.slider("최대 풍속 크기 ($V_{max}$, m/s)", 5, 80, 40, step=5)
        v_scale = st.slider("기압계 반경 크기 ($R_{scale}$, km)", 100, 1000, 400, step=50)
        
        # 제어 변수 다양화
        C_d_coef = st.slider(
            "지표 항력 계수 ($C_d$)", 0.0005, 0.0050, 0.0015, step=0.0005, format="%.4f",
            help="지표면의 마찰 특성에 따른 무차원 항력 계수입니다. 지표가 거칠수록 응력이 커집니다."
        )
        
        inflow_angle = st.slider(
            "마찰 수렴각 (Inflow Angle, °)", 0, 45, 25, step=5,
            help="지상풍이 저기압 중심으로 꺾여 들어오는(혹은 고기압에서 나가는) 각도입니다."
        )
        
        eccentricity = st.slider(
            "기압계 타원형 이심률 ($e$)", 0.0, 0.9, 0.0, step=0.1,
            help="기압계의 장축과 단축 비를 조절하여 비대칭적인 에크만 펌핑 형태를 재현합니다."
        )

    # 2D 격자 및 물리 연산 생성
    Grid_L = 1600 * 1000  # 영역 크기 (1600 km)
    Grid_N = 60           # 해상도
    x_g = np.linspace(-Grid_L / 2, Grid_L / 2, Grid_N)
    y_g = np.linspace(-Grid_L / 2, Grid_L / 2, Grid_N)
    X_g, Y_g = np.meshgrid(x_g, y_g)

    # 타원형 좌표 변환 적용 (이심률 반영)
    # x축 방향을 단축, y축 방향을 장축으로 비대칭 팽창
    aspect_ratio_y = 1.0 / np.sqrt(1.0 - eccentricity**2) if eccentricity < 1.0 else 1.0
    R_g = np.sqrt(X_g**2 + (Y_g / aspect_ratio_y)**2) + 1e-5
    theta_g = np.arctan2(Y_g / aspect_ratio_y, X_g)

    R_scale_m = v_scale * 1000
    
    # 가우시안 풍속 전개
    v_mag = max_wind_speed * (R_g / R_scale_m) * np.exp(-((R_g / R_scale_m) ** 2))
    alpha_rad = np.radians(inflow_angle)

    if "저기압" in pressure_system or "기압골" in pressure_system:
        # 반시계 방향으로 수렴 (내향각 alpha 적용)
        u_wind = -v_mag * np.sin(theta_g + alpha_rad)
        v_wind = v_mag * np.cos(theta_g + alpha_rad)
    else:
        # 고기압: 시계 방향으로 발산 (외향각 alpha 적용)
        u_wind = v_mag * np.sin(theta_g - alpha_rad)
        v_wind = -v_mag * np.cos(theta_g - alpha_rad)

    # 마찰 응력 수식: Tau = rho * Cd * |V| * V
    wind_speed = np.sqrt(u_wind**2 + v_wind**2) + 1e-5
    tau_x_mat = rho_a * C_d_coef * wind_speed * u_wind
    tau_y_mat = rho_a * C_d_coef * wind_speed * v_wind

    # 유한차분법(FDM) 중앙차분(Central Difference)으로 curl(Tau) 연산
    dx_g = x_g[1] - x_g[0]
    dy_g = y_g[1] - y_g[0]

    dtauy_dx_g = np.gradient(tau_y_mat, axis=1) / dx_g
    dtaux_dy_g = np.gradient(tau_x_mat, axis=0) / dy_g
    curl_tau_g = dtauy_dx_g - dtaux_dy_g

    # 에크만 연직 속도 계산 (w_e = curl(tau) / (rho * f)) -> cm/s 단위 변환
    w_e_val = (curl_tau_g / (rho_a * f)) * 100.0

    with col2:
        # Plotly Contour 및 Vector Cone 시각화
        fig_b = gr.Figure()

        # 1. 2D 컬러 등고선 평면 (연직 상승/하강 속도)
        fig_b.add_trace(
            gr.Contour(
                z=w_e_val,
                x=x_g / 1000,
                y=y_g / 1000,
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="연직 속도 $w_e$ (cm/s)"),
                contours=dict(showlines=True),
                name="Ekman Vertical Motion"
            )
        )

        # 2. 입체적 3D 원뿔 화살표장 중첩
        skip_n = 4
        fig_b.add_trace(
            gr.Cone(
                x=X_g[::skip_n, ::skip_n].flatten() / 1000,
                y=Y_g[::skip_n, ::skip_n].flatten() / 1000,
                z=np.zeros_like(X_g[::skip_n, ::skip_n].flatten()),
                u=u_wind[::skip_n, ::skip_n].flatten(),
                v=v_wind[::skip_n, ::skip_n].flatten(),
                w=w_e_val[::skip_n, ::skip_n].flatten() / 5.0,  # 가독성을 위한 크기 매핑 비율 조정
                sizemode="scaled",
                sizeref=2.5,
                colorscale="Cividis",
                showscale=False,
                name="Surface Wind Field"
            )
        )

        fig_b.update_layout(
            title="에크만 펌핑 연직 속도($w_e$) 분포 및 지상풍 벡터장 (FDM 연산)",
            xaxis_title="동서 X 방향 거리 (km)",
            yaxis_title="남북 Y 방향 거리 (km)",
            height=600,
        )
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown(
        f"""
        ---
        ### 📝 모듈 B 동역학적 핵심 메커니즘 해설 (가독성 요약)
        *   **제어 변수의 유기적 영향성**:
            *   **항력 계수($C_d$)의 확장**: 지표 응력의 강도를 제어합니다. 항력 계수가 높아질수록 난류 경계층 마찰 응력이 증폭되어 수렴/발산 흐름이 강해지고, 최종 연직 속도($w_e$) 역시 정비례하여 증가합니다.
            *   **마찰 수렴각($\\alpha$)**: 지균 대칭 상태에서 벗어나는 마찰적 내향 수렴의 정도를 나타내며, 수렴각이 클수록 에크만 수송에 의한 수평 질량 축적이 급격해집니다.
            *   **이심률($e$)에 따른 비대칭 구조**: 타원형 기압 배치 모델을 구동하면 등압선의 곡률 변동성이 연직 기류의 비대칭적 집중(예: 기압골 축을 따른 선형 상승류 띠 형성)을 어떻게 야기하는지 정밀 시각화할 수 있습니다.
        *   **기압계 수명 조절**: 
            *   **저기압성 순환**: 시계 반대 방향 수렴을 유도하고 중심부에서 **강한 상승 기류($w_e > 0$)**를 형성하여 구름과 구름 대류계를 생성합니다.
            *   **고기압성 순환**: 시계 방향 발산 흐름으로 중심부에 **하강 기류($w_e < 0$)**를 발생시켜 맑고 건조한 대기 안정도를 자아냅니다.
        """
    )

# ==============================================================================
# MODULE C: NONLINEAR GRADIENT WIND ADJUSTMENT (NEW CONCEPT)
# ==============================================================================
with tab3:
    st.header("모듈 C: 비선형 경도풍 적응 및 마찰 감쇄 궤적 (Trajectory Simulator)")
    st.markdown(
        """
        ### 📌 개요 및 지배 방정식
        기존 지구과학 II 교과서는 대기 운동을 '직선 등압선(지균풍)' 또는 '완전 평형 상태(경도풍)'로만 이분화하여 설명합니다. 
        본 모듈은 **곡률이 존재하는 원형 등압선 내에서 출발한 공기 덩어리가 전향력, 원심력, 기압경도력, 선형 마찰력의 상호작용 하에 평형을 찾아가는 비선형 동적 시간 변화**를 전면 시뮬레이션합니다.
        
        공기 덩어리 궤적 추적용 라그랑주적 지배 방정식 (극좌표 기반 물리 수치 모델):
        
        $$\\frac{du}{dt} = f v + \\frac{v^2}{R} + P_{grad} - r u$$
        $$\\frac{dv}{dt} = -f u - \\frac{u v}{R} - r v$$
        
        *   $P_{grad} = -\\frac{1}{\\rho_a}\\frac{\\partial P}{\\partial r}$ :动徑 방향 기압 경도력  
        *   $\\frac{v^2}{R}$ 및 $\\frac{u v}{R}$ : 곡률 반경 $R$에 따른 **원심력 및 겉보기 전향 보정 가속도**
        *   $-r u, -r v$ : **선형 지표 마찰력** (마찰 계수 $r$)
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 모듈 C 동적 설정 변수")
        test_case = st.selectbox(
            "시뮬레이션 가동 시나리오 선택",
            [
                "급격한 기압 변화에 따른 관성 진동 (Inertial Oscillation)",
                "마찰력에 의한 저기압성 경도풍 수렴 (Cyclonic Adjustment)",
                "고기압 흐름에서의 원심력 가속 현상 (Anticyclonic Supergeostrophic)"
            ]
        )
        
        # 기압 경도력 및 초기 풍속 설정
        p_grad_c = st.slider("기압경도력 크기 ($P_{grad}$, m/s²)", 1e-4, 15e-4, 6e-4, format="%.1e")
        r_fric = st.slider("공기 분자 마찰계수 ($r$, s⁻¹)", 0.0, 3e-4, 5e-5, format="%.1e")
        curv_radius = st.slider("등압선 곡률 반경 ($R$, km)", 100, 2000, 800, step=100)
        
        init_u = st.slider("초기 동서풍속 ($u_0$, m/s)", -20.0, 40.0, 0.0, step=1.0)
        init_v = st.slider("초기 남북풍속 ($v_0$, m/s)", -20.0, 40.0, 10.0, step=1.0)
        
        t_duration = st.slider("수치 시뮬레이션 시간 (시간, hours)", 12, 120, 48, step=6)

    # 곡률 반경 m 단위 변환
    R_m = curv_radius * 1000.0

    # 비선형 상미분 방정식계 (ODE) 정의
    def gradient_wind_ode(t, state, f_val, p_g, r_val, R_val):
        u, v = state
        # 극좌표 가속도 항과 원심력 효과 반영
        du_dt = f_val * v + (v**2 / R_val) + p_g - r_val * u
        dv_dt = -f_val * u - (u * v / R_val) - r_val * v
        return [du_dt, dv_dt]

    # IVP 수치 해석 기법 (RK45) 적용
    t_span = (0, t_duration * 3600)
    t_eval = np.linspace(0, t_duration * 3600, 1500)
    init_state = [init_u, init_v]

    sol_c = solve_ivp(
        gradient_wind_ode,
        t_span,
        init_state,
        args=(f, p_grad_c, r_fric, R_m),
        t_eval=t_eval,
        method="RK45"
    )

    with col2:
        if sol_c.success:
            u_traj = sol_c.y[0]
            v_traj = sol_c.y[1]
            time_h = sol_c.t / 3600.0

            # 2차원 위상 공간(Phase Space) 및 궤적선 드로잉
            fig_c = gr.Figure()

            # 1. 덩어리 속도 벡터의 경로 시각화
            fig_c.add_trace(
                gr.Scatter(
                    x=u_traj,
                    y=v_traj,
                    mode="lines",
                    line=dict(color="royalblue", width=3),
                    name="바람 궤적선"
                )
            )

            # 시간에 따른 궤적 위치 마커 표시
            fig_c.add_trace(
                gr.Scatter(
                    x=u_traj,
                    y=v_traj,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=time_h,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="경과 시간 (Hours)", x=1.05)
                    ),
                    name="시간별 추적 점"
                )
            )

            # 시작점 표시
            fig_c.add_trace(
                gr.Scatter(
                    x=[init_u],
                    y=[init_v],
                    mode="markers",
                    marker=dict(color="green", size=12, symbol="triangle-up"),
                    name="초기 운동 상태"
                )
            )

            # 최종 수렴점 (마지막 스텝 상태)
            fig_c.add_trace(
                gr.Scatter(
                    x=[u_traj[-1]],
                    y=[v_traj[-1]],
                    mode="markers",
                    marker=dict(color="red", size=14, symbol="star-cross"),
                    name="수렴 조절점 (Steady State)"
                )
            )

            fig_c.update_layout(
                title="비선형 풍속 위상 다이어그램 (Velocity Hodograph Space)",
                xaxis_title="U 성분 풍속 (m/s)",
                yaxis_title="V 성분 풍속 (m/s)",
                height=550,
            )
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.error("풍속 궤적 적분 연산에 오류가 발생했습니다.")

    st.markdown(
        f"""
        ---
        ### 📝 모듈 C 동역학적 핵심 메커니즘 해설 (가독성 요약)
        *   **원심력 개입에 따른 비선형 적응 메커니즘**:
            *   직선 등압선과 달리 곡률 등압선 내 운동은 **원심력($V^2/R$)**이 추가 지배 물리로 개입합니다. 이로 인해 궤적이 원형 회전 운동을 하며 일그러지는 **진동 수렴 구조**를 관찰할 수 있습니다.
            *   **마찰력($r = 0$)이 전혀 없을 때**: 외부 기압 변화 시 진동이 상쇄되지 못하고 지속적인 폐곡선을 그리는 **무한 관성 진동(Inertial Oscillation)**이 발생합니다.
            *   **마찰력($r > 0$)이 작동할 때**: 마찰력이 지속적인 감쇄(Damping) 작용을 하여 위상 다이어그램상에서 나선형으로 수축하며 **점근적인 긍정적 경도풍 평형 상태(Steady State)**로 수렴합니다.
        *   **세특 기록용 핵심 관전 포인트**: 초기 속도가 0인 상태에서 급격한 기압 경도력이 인가될 때 코리올리 힘과 마찰력이 바람을 나선형 궤도로 소용돌이치게 만드는데, 이 동역학적 천이 과정을 Runge-Kutta(RK45) 기법으로 추적하는 동역학 연구 사례로 서술하기 최적의 조건입니다.
        """
    )
