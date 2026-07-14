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
# MODULE A: EKMAN SPIRAL (BVP SOLVER) - 거칠기 효과 극대화 및 수치적 안정화
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
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 모듈 A 제어 변수")
        system_type = st.radio("물리계 선택", ["대기 경계층 (Atmosphere)", "해양 혼합층 (Ocean)"])
        
        K_m = st.slider(
            "연직 에디 점성 계수 ($K_m$, m²/s)", 0.5, 30.0, 5.0, step=0.5,
            help="공기의 난류 점성을 나타내며, 이 값이 클수록 마찰 영향이 상층까지 강하게 전달됩니다."
        )
        
        z_0_preset = st.selectbox(
            "지표 거칠기 유형 ($z_0$ 사전 설정)",
            ["매끄러운 지표면 (0.01m)", "낮은 수목 지대 (0.1m)", "숲 및 도시 지역 (1.0m)", "고층 빌딩 밀집 지역 (3.0m)"]
        )
        
        if "매끄러운" in z_0_preset:
            z_0_val = 0.01
        elif "수목" in z_0_preset:
            z_0_val = 0.1
        elif "숲" in z_0_preset:
            z_0_val = 1.0
        else:
            z_0_val = 3.0
            
        ug_val = st.slider("자유 대기 지균풍 ($u_g$, m/s)", -25.0, 25.0, 15.0, step=1.0, key="modA_ug")
        vg_val = st.slider("자유 대기 지균풍 ($v_g$, m/s)", -25.0, 25.0, 0.0, step=1.0, key="modA_vg")
        z_max = st.slider("관측 최대 고도 ($z_{max}$, m)", 200, 2000, 1000, step=100, key="modA_zmax")

    # 수치적 안정을 위해 로그 그리드 대신 충분한 밀도의 균일 선형 그리드 사용
    z_steps = np.linspace(z_0_val, z_max, 200)
    y_guess = np.zeros((4, z_steps.size))

    def ekman_bvp_sys(z, y):
        u, dudz, v, dvdz = y
        d2u_dz2 = -f * (v - vg_val) / K_m
        d2v_dz2 = f * (u - ug_val) / K_m
        return [dudz, d2u_dz2, dvdz, d2v_dz2]

    def ekman_bc(ya, yb):
        if system_type == "대기 경계층 (Atmosphere)":
            return [ya[0], ya[2], yb[0] - ug_val, yb[2] - vg_val]
        else:
            return [ya[0] - ug_val, ya[2] - vg_val, yb[0], yb[2]]

    res_bvp = solve_bvp(ekman_bvp_sys, ekman_bc, z_steps, y_guess)

    with col2:
        if res_bvp.success:
            z_sol = res_bvp.x
            u_sol = res_bvp.y[0]
            v_sol = res_bvp.y[2]

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
            st.error("BVP 수치 해석 솔버가 수렴에 실패했습니다. 입력 매개변수를 조정해 주세요.")

    st.markdown(
        f"""
        ---
        ### 📝 모듈 A 동역학적 핵심 메커니즘 해설 (가독성 요약)
        *   **지표 거칠기 길이($z_0$)의 파급 효과**: 
            *   **$z_0$가 작을 때(예: 매끄러운 지표면)**: 마찰 효과가 매우 얕은 고도에서 소멸하여 바람이 고도 상승에 따라 지균풍 균형($u_g$, $v_g$)으로 빠르게 복귀합니다.
            *   **$z_0$가 클 때(예: 대도시 빌딩숲)**: 난류 경계층의 거칠기로 인해 저항 마찰력이 상층 흐름에 미치는 영향이 비약적으로 증가합니다. 이에 따라 지상 근처의 풍속 감소 효과가 강해지며 지상풍 방향이 지균풍 방향에 비해 크게 꺾입니다.
        """
    )

# ==============================================================================
# MODULE B: EKMAN PUMPING (FDM VORTICITY SOLVER) - 다양성 확장
# ==============================================================================
with tab2:
    st.header("모듈 B: 에크만 펌핑(Ekman Pumping)과 복합 기압 배치 시뮬레이터")
    st.markdown(
        """
        ### 📌 개요 및 지배 방정식
        경계층 내 마찰에 의해 지상풍이 저기압 중심으로 수렴하거나 고기압 중심에서 발산할 때 연직 운동이 강제됩니다.
        
        $$w_e = \\frac{1}{\\rho_a f} \\nabla \\times \\vec{\\tau} = \\frac{1}{\\rho_a f} \\left( \\frac{\\partial \\tau_y}{\\partial x} - \\frac{\\partial \\tau_x}{\\partial y} \\right)$$
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 모듈 B 기압계 제어 변수")
        pressure_system = st.selectbox(
            "기압 배치 시스템 선택",
            ["원형 저기압 (Typhoon)", "원형 고기압 (Anticyclone)", "타원형 기압골 (Elliptical Trough)"]
        )
        
        max_wind_speed = st.slider("최대 풍속 크기 ($V_{max}$, m/s)", 5, 80, 40, step=5, key="modB_vmax")
        v_scale = st.slider("기압계 반경 크기 ($R_{scale}$, km)", 100, 1000, 400, step=50, key="modB_vscale")
        
        C_d_coef = st.slider(
            "지표 항력 계수 ($C_d$)", 0.0005, 0.0050, 0.0015, step=0.0005, format="%.4f",
            help="지표 마찰 특성에 따른 항력 계수입니다."
        )
        
        inflow_angle = st.slider(
            "마찰 수렴각 (Inflow Angle, °)", 0, 45, 25, step=5,
            help="지상풍이 저기압 중심으로 꺾여 들어오는 각도입니다."
        )
        
        eccentricity = st.slider(
            "기압계 타원형 이심률 ($e$)", 0.0, 0.9, 0.0, step=0.1,
            help="기압계의 장축과 단축 비를 조절하여 대칭성을 제어합니다."
        )

    # 2D 격자 및 물리 연산 생성
    Grid_L = 1600 * 1000  
    Grid_N = 60           
    x_g = np.linspace(-Grid_L / 2, Grid_L / 2, Grid_N)
    y_g = np.linspace(-Grid_L / 2, Grid_L / 2, Grid_N)
    X_g, Y_g = np.meshgrid(x_g, y_g)

    aspect_ratio_y = 1.0 / np.sqrt(1.0 - eccentricity**2) if eccentricity < 1.0 else 1.0
    R_g = np.sqrt(X_g**2 + (Y_g / aspect_ratio_y)**2) + 1e-5
    theta_g = np.arctan2(Y_g / aspect_ratio_y, X_g)

    R_scale_m = v_scale * 1000
    v_mag = max_wind_speed * (R_g / R_scale_m) * np.exp(-((R_g / R_scale_m) ** 2))
    alpha_rad = np.radians(inflow_angle)

    if "저기압" in pressure_system or "기압골" in pressure_system:
        u_wind = -v_mag * np.sin(theta_g + alpha_rad)
        v_wind = v_mag * np.cos(theta_g + alpha_rad)
    else:
        u_wind = v_mag * np.sin(theta_g - alpha_rad)
        v_wind = -v_mag * np.cos(theta_g - alpha_rad)

    wind_speed = np.sqrt(u_wind**2 + v_wind**2) + 1e-5
    tau_x_mat = rho_a * C_d_coef * wind_speed * u_wind
    tau_y_mat = rho_a * C_d_coef * wind_speed * v_wind

    dx_g = x_g[1] - x_g[0]
    dy_g = y_g[1] - y_g[0]

    dtauy_dx_g = np.gradient(tau_y_mat, axis=1) / dx_g
    dtaux_dy_g = np.gradient(tau_x_mat, axis=0) / dy_g
    curl_tau_g = dtauy_dx_g - dtaux_dy_g

    w_e_val = (curl_tau_g / (rho_a * f)) * 100.0

    with col2:
        fig_b = gr.Figure()
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

        # 버전에 영향이 없고 실시간 2D 평면 화살표를 가장 안전하게 시각화할 수 있는 Cone 3D/2.D 겸용 구조
        skip_n = 4
        fig_b.add_trace(
            gr.Cone(
                x=X_g[::skip_n, ::skip_n].flatten() / 1000,
                y=Y_g[::skip_n, ::skip_n].flatten() / 1000,
                z=np.zeros_like(X_g[::skip_n, ::skip_n].flatten()),
                u=u_wind[::skip_n, ::skip_n].flatten(),
                v=v_wind[::skip_n, ::skip_n].flatten(),
                w=w_e_val[::skip_n, ::skip_n].flatten() / 5.0,  
                sizemode="scaled",
                sizref=2.5,
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
        *   **지표 마찰계수와 풍속의 비선형 관계**: 기압 중심부 부근의 풍속이 강할수록 풍응력($\\vec{\\tau}$)은 풍속의 제곱($V^2$)에 비례하여 강해집니다. 따라서 기압계 반경과 최대 풍속의 미세한 증가가 중심 연직 흐름을 폭발적으로 증가시킵니다.
        *   **타원 형태 이심률의 비대칭성**: 타원형 기압골 구조를 선택하면 곡률 반경이 축에 따라 달라져, 원형 대칭 구조와 달리 특정 구역에 정렬된 띠(Band) 모양의 매우 강한 상승 기류 구역이 형성됩니다.
        """
    )

# ==============================================================================
# MODULE C: TIME-DEPENDENT WIND ADJUSTMENT (완전 신개념 재설계)
# ==============================================================================
with tab3:
    st.header("모듈 C: 시간에 따른 경도풍(Gradient Wind) 비선형 조절 및 마찰 감쇄 궤적")
    st.markdown(
        """
        ### 📌 개요 및 지배 방정식
        고등학교 지구과학 II에서는 항상 힘의 평형 상태인 '정적 상태'만 다룹니다.
        본 모듈은 **기압 변화가 유발되었을 때 공기가 원형 등압선 내에서 전향력, 원심력, 기압경도력, 마찰력을 상호 작용시키며 시간의 흐름에 따라 최종 균형(경도풍)으로 조절되는 물리 과정**을 시간 의존성(Time-dependent) 상미분 방정식으로 풀어냅니다.
        
        기압 중심과의 거리($R$)를 유지하는 등압선 모델 상에서 공기 덩어리의 가속도 방정식:
        
        $$\\frac{du}{dt} = f v + \\frac{v^2}{R} - r u$$
        $$\\frac{dv}{dt} = -f u - \\frac{u v}{R} - P_{grad} - r v$$
        
        *   $P_{grad} = \\frac{1}{\\rho_a}\\frac{\\partial P}{\\partial r}$ :動徑(반지름) 방향 기압경도력
        *   $\\frac{v^2}{R}$ : 운동 궤적이 만곡함에 따라 추가되는 실시간 **원심력 효과**
        *   $-r u, -r v$ : 운동 속도에 비례하여 작용하는 지표 선형 마찰력 (마찰 감쇄 계수 $r$)
        """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 모듈 C 동적 설정 변수")
        system_type_c = st.radio("기압 조건", ["저기압성 순환 (Cyclonic)", "고기압성 순환 (Anticyclonic)"])
        
        # 물리 파라미터 조절기
        p_grad_c = st.slider("기압경도력 가속도 ($P_{grad}$, m/s²)", 1e-4, 15e-4, 5e-4, format="%.1e", key="modC_pgrad")
        r_fric = st.slider("마찰 감쇄 계수 ($r$, s⁻¹)", 0.0, 3e-4, 5e-5, format="%.1e", key="modC_rfric")
        curv_radius = st.slider("등압선 곡률 반경 ($R$, km)", 200, 2000, 800, step=100, key="modC_R")
        
        st.markdown("**공기 덩어리 초기 속도 상태**")
        init_u = st.slider("초기 U 성분 속도 (m/s)", -30.0, 30.0, 0.0, step=1.0, key="modC_initu")
        init_v = st.slider("초기 V 성분 속도 (m/s)", -30.0, 30.0, 0.0, step=1.0, key="modC_initv")
        
        t_duration = st.slider("궤적 추적 시간 (Simulation Time, Hours)", 12, 120, 48, step=6, key="modC_tdur")

    # M 단위 환산
    R_m = curv_radius * 1000.0

    # 비선형 경도풍 상미분 방정식 시스템 정의
    def gradient_wind_ode(t, state, f_val, p_g, r_val, R_val, sys_type):
        u, v = state
        
        # 저기압성과 고기압성에 따른 힘의 부호 결정
        if sys_type == "저기압성 순환 (Cyclonic)":
            # 저기압: 기압경도력이 안쪽(-방향), 원심력은 바깥쪽(+방향)
            du_dt = f_val * v + (v**2 / R_val) - r_val * u
            dv_dt = -f_val * u - (u * v / R_val) - p_g - r_val * v
        else:
            # 고기압: 기압경도력이 바깥쪽(+방향), 원심력도 바깥쪽(+방향), 전향력이 안쪽(-방향)
            du_dt = f_val * v + (v**2 / R_val) - r_val * u
            dv_dt = -f_val * u - (u * v / R_val) + p_g - r_val * v
            
        return [du_dt, dv_dt]

    # Runge-Kutta 45 수치 적분 수행
    t_span = (0, t_duration * 3600)
    t_eval = np.linspace(0, t_duration * 3600, 1000)
    init_state = [init_u, init_v]

    sol_c = solve_ivp(
        gradient_wind_ode,
        t_span,
        init_state,
        args=(f, p_grad_c, r_fric, R_m, system_type_c),
        t_eval=t_eval,
        method="RK45"
    )

    with col2:
        if sol_c.success:
            u_traj = sol_c.y[0]
            v_traj = sol_c.y[1]
            time_h = sol_c.t / 3600.0

            # 2차원 풍속 평면(Hodograph) 시각화
            fig_c = gr.Figure()

            # 1. 속도 공간 안에서의 점진적 진동 궤적
            fig_c.add_trace(
                gr.Scatter(
                    x=u_traj,
                    y=v_traj,
                    mode="lines",
                    line=dict(color="blue", width=3),
                    name="수치해석 바람 경로"
                )
            )

            # 시간에 따른 궤적의 이동 방향 시각화
            fig_c.add_trace(
                gr.Scatter(
                    x=u_traj,
                    y=v_traj,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=time_h,
                        colorscale="Jet",
                        showscale=True,
                        colorbar=dict(title="경과 시간 (Hours)", x=1.05)
                    ),
                    name="시간적 추적 포인트"
                )
            )

            # 시작점 및 종착 평형점 표기
            fig_c.add_trace(
                gr.Scatter(
                    x=[init_u],
                    y=[init_v],
                    mode="markers",
                    marker=dict(color="green", size=12, symbol="circle"),
                    name="초기 기동 속도"
                )
            )

            fig_c.add_trace(
                gr.Scatter(
                    x=[u_traj[-1]],
                    y=[v_traj[-1]],
                    mode="markers",
                    marker=dict(color="red", size=14, symbol="star"),
                    name="최종 수렴 경도풍 평형점"
                )
            )

            # 중심 지표면 마찰 수렴선 가시화 보조
            fig_c.update_layout(
                title=f"비선형 풍속 위상 상태 다이어그램 (Hodograph) - {system_type_c}",
                xaxis_title="동서 풍속 U (m/s)",
                yaxis_title="남북 풍속 V (m/s)",
                height=550,
            )
            st.plotly_chart(fig_c, use_container_width=True)
            
            # 최종 수렴 상태 정량 분석 요약
            st.success(
                f"🎯 **동역학 시뮬레이션 결과 요약:** 설정된 원형 기압 시스템 하에서 공기 덩어리는 관성 진동을 거치며 "
                f"최종 정상류 평형 상태인 **U = {u_traj[-1]:.2f} m/s, V = {v_traj[-1]:.2f} m/s**로 점근적 안착(감쇄 조절)을 마쳤습니다."
            )
        else:
            st.error("풍속 조절 궤적 적분 연산 과정에서 수치 오류가 발생했습니다.")

    st.markdown(
        f"""
        ---
        ### 📝 모듈 C 동역학적 핵심 메커니즘 해설 (가독성 요약)
        *   **원심력 개입에 따른 비선형 적응 메커니즘**:
            *   등압선이 원형으로 휘어져 있을 때, 속도가 빨라질수록 바깥쪽으로 나가려는 **원심력($V^2/R$)**이 동적으로 변동합니다. 이에 따라 직선 지균 조절에서보다 훨씬 복잡하고 일그러진 타원형 수렴 곡선을 그립니다.
            *   **마찰력($r = 0$)이 작동하지 않을 때 (무마찰 등압선)**: 공기 덩어리는 기압경도력과 전향력의 불균형으로 인해 평형점에 안착하지 못하고, 평형점을 중심으로 **지속적으로 공전하는 등진동 관성 회동(Inertial Oscillation)**을 유지합니다.
            *   **마찰력($r > 0$)이 가동될 때**: 저항 마찰력이 공기의 에너지를 지속적으로 뺏어가므로 회전 반경이 시시각각 수축(Damping)하며, 결국 나선형 궤적을 그리며 **수렴 경도풍 평형 상태(Steady-state Gradient Wind)**에 도달하게 됩니다.
        """
    )
