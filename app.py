import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap

from dqn_trainer import train_dqn, get_dqn_path
from comparison import compare_algorithms


st.set_page_config(
    page_title="MazeMind RL | RL Lab",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --mm-bg:#091426; --mm-panel:#0d1d34; --mm-panel2:#10233e; --mm-line:#244364; --mm-text:#eef5ff; --mm-muted:#9fb1c9; --mm-coral:#ff6f61; }
.stApp { background:radial-gradient(ellipse at 70% -10%,#142a48 0%,#081323 55%); color:var(--mm-text); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(8,19,35,.86); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#07152a,#0c1b30); border-right:1px solid var(--mm-line); }
[data-testid="stSidebar"] > div { padding-top:1.2rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#f5f8ff !important; letter-spacing:-.025em; }
h1 { font-size:2.5rem !important; }
p,label,.stCaption { color:#bdcbe0; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#122642,#0c1a2e); border:1px solid #244364; border-radius:14px; padding:13px 16px; min-height:105px; }
[data-testid="stMetricLabel"] { color:#a9bbd2 !important; }
[data-testid="stMetricValue"] { color:#f6f8ff !important; font-family:'Space Grotesk',sans-serif; }
.stButton > button { border-radius:10px; border:1px solid #ff8b7f; background:#ff6f61; color:#171d2a; font-weight:700; min-height:2.6rem; }
.stButton > button:hover { background:#ff9388; border-color:#ffb3aa; color:#171d2a; }
hr { border-color:#244364; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stDataFrame"] { border:1px solid #244364; border-radius:12px; overflow:hidden; }
[data-testid="stExpander"] { border:1px solid #244364; border-radius:14px; background:rgba(13,29,52,.7); }
[data-testid="stPlotlyChart"], [data-testid="stImage"] { border-radius:12px; }
.small-label { color:#9fb1c9; font-size:.76rem; text-transform:uppercase; letter-spacing:.13em; font-weight:700; }
.hero { padding:.25rem 0 .45rem; }
.hero-tag { display:inline-block; padding:5px 10px; border:1px solid #27516a; border-radius:99px; color:#8ce8d4; background:#102c3b; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.hero-sub { color:#a9bbd2; font-size:1rem; margin-top:-.5rem; }
.panel-title { font-size:1.15rem; font-weight:700; color:#f4f7ff; }
div[data-testid="stVerticalBlock"] div:has(> .metric-accent) { border-radius:12px; }
</style>
""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><span class="hero-tag">Reinforcement learning lab</span><h1>MazeMind <span style="color:#ff765f">RL</span></h1><p class="hero-sub">Train an agent. Explore its strategy. Compare the algorithms.</p></div>""", unsafe_allow_html=True)


# 0 = free cell, 1 = wall
MAZE = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
], dtype=int)

ROWS, COLS = MAZE.shape
START = (0, 0)
GOAL = (7, 7)

# Up, right, down, left
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def state_id(position):
    return position[0] * COLS + position[1]


def move(position, action):
    dr, dc = ACTIONS[action]
    nr = position[0] + dr
    nc = position[1] + dc

    # Stay in place when hitting a boundary or wall
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return position, -1.0, False

    if MAZE[nr, nc] == 1:
        return position, -1.0, False

    next_position = (nr, nc)

    if next_position == GOAL:
        return next_position, 10.0, True

    return next_position, -0.05, False


def train_agent(episodes, alpha, gamma, epsilon):
    q_table = np.zeros((ROWS * COLS, len(ACTIONS)))
    rewards = []
    successes = []

    progress = st.progress(0, text="Training the robot...")

    for episode in range(episodes):
        position = START
        total_reward = 0.0
        success = False

        # Gradually explore less as learning progresses
        explore = max(0.02, epsilon * (1 - episode / episodes))

        for _ in range(150):
            s = state_id(position)

            if np.random.random() < explore:
                action = np.random.randint(len(ACTIONS))
            else:
                action = int(np.argmax(q_table[s]))

            next_position, reward, done = move(position, action)
            ns = state_id(next_position)

            # Q-learning update
            target = reward
            if not done:
                target += gamma * np.max(q_table[ns])

            q_table[s, action] += alpha * (
                target - q_table[s, action]
            )

            position = next_position
            total_reward += reward

            if done:
                success = True
                break

        rewards.append(total_reward)
        successes.append(int(success))

        if (episode + 1) % max(1, episodes // 100) == 0:
            progress.progress(
                (episode + 1) / episodes,
                text=f"Training episode {episode + 1}/{episodes}",
            )

    progress.empty()
    return q_table, rewards, successes


def get_path(q_table, max_steps=100):
    position = START
    path = [position]
    visited = {position}

    for _ in range(max_steps):
        action = int(np.argmax(q_table[state_id(position)]))
        next_position, _, done = move(position, action)

        if next_position == position:
            break

        position = next_position
        path.append(position)

        if done:
            break

        if position in visited:
            break

        visited.add(position)

    return path


def draw_maze(path, compact=False):
    board = MAZE.copy()
    board[START] = 2
    board[GOAL] = 3
    for r, c in path:
        if (r, c) not in (START, GOAL):
            board[r, c] = 4

    cmap = ListedColormap(["#f2f5fa", "#22334b", "#4b9ce8", "#2fbd87", "#ffd166"])
    size = 4.25 if compact else 4.8
    fig, ax = plt.subplots(figsize=(size, size), facecolor="#0d1d34")
    ax.set_facecolor("#0d1d34")
    ax.imshow(board, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))
    ax.set_xticklabels(range(COLS), color="#9fb1c9", fontsize=8)
    ax.set_yticklabels(range(ROWS), color="#9fb1c9", fontsize=8)
    ax.set_xticks(np.arange(-.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="#9aaec4", linewidth=.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, pad=4)
    ax.text(START[1], START[0], "S", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center", fontsize=12, fontweight="bold", color="#063629")
    ax.set_title("LEARNED ROUTE", color="#eaf2ff", fontsize=10, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_color("#244364")
    fig.tight_layout(pad=.6)
    return fig


# Sidebar controls
st.sidebar.markdown("### 🧪 Training controls")
st.sidebar.caption("Configure an experiment, then train the agent.")

algorithm = st.sidebar.selectbox(
    "Choose RL algorithm",
    ["Q-Learning", "Deep Q-Network (DQN)"],
)

episodes = st.sidebar.slider(
    "Training episodes", 100, 5000, 1500, 100
)

if algorithm == "Q-Learning":
    alpha = st.sidebar.slider(
        "Learning rate", 0.01, 1.0, 0.15
    )
    gamma = st.sidebar.slider(
        "Discount factor", 0.1, 0.99, 0.95
    )
    epsilon = st.sidebar.slider(
        "Initial exploration", 0.1, 1.0, 1.0
    )
else:
    alpha = st.sidebar.slider(
        "DQN learning rate",
        0.0001,
        0.01,
        0.001,
        format="%.4f",
    )
    gamma = st.sidebar.slider(
        "DQN discount factor", 0.1, 0.99, 0.95
    )
    epsilon = st.sidebar.slider(
        "DQN initial exploration", 0.1, 1.0, 1.0
    )


# Initialize all session-state values independently
defaults = {
    "q_table": None,
    "dqn_agent": None,
    "rewards": [],
    "successes": [],
    "dqn_losses": [],
    "trained_settings": None,
    "trained_algorithm": None,
    "previous_run": None,
    "comparison_results": None,
    "last_training_time": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Train the selected algorithm
if st.sidebar.button("🧠 Train robot", type="primary"):
    training_started = time.perf_counter()
    if st.session_state.trained_algorithm is not None:
        st.session_state.previous_run = {
            "algorithm": st.session_state.trained_algorithm,
            "settings": st.session_state.trained_settings,
            "rewards": st.session_state.rewards.copy(),
            "successes": st.session_state.successes.copy(),
        }

    if algorithm == "Q-Learning":
        q, rewards, successes = train_agent(
            episodes, alpha, gamma, epsilon
        )

        st.session_state.q_table = q
        st.session_state.dqn_agent = None
        st.session_state.dqn_losses = []

    else:
        progress = st.progress(0, text="Training DQN agent...")

        def update_progress(fraction, episode, total):
            progress.progress(
                fraction,
                text=f"DQN training episode {episode}/{total}",
            )

        agent, rewards, successes, losses = train_dqn(
            episodes=episodes,
            learning_rate=alpha,
            gamma=gamma,
            epsilon=epsilon,
            progress_callback=update_progress,
        )

        progress.empty()
        st.session_state.dqn_agent = agent
        st.session_state.q_table = None
        st.session_state.dqn_losses = losses

    st.session_state.rewards = rewards
    st.session_state.successes = successes
    st.session_state.trained_algorithm = algorithm
    st.session_state.trained_settings = {
        "Algorithm": algorithm,
        "Episodes": episodes,
        "Learning rate": alpha,
        "Discount factor": gamma,
        "Initial exploration": epsilon,
    }

    st.session_state.last_training_time = time.perf_counter() - training_started
    st.success(f"{algorithm} training completed!")


# Main overview: compact metric cards at the top
st.markdown("<div class='small-label'>EXPERIMENT OVERVIEW</div>", unsafe_allow_html=True)
if st.session_state.trained_algorithm is not None:
    rewards = st.session_state.rewards
    successes = st.session_state.successes
    top_success = 100 * float(np.mean(successes[-100:])) if successes else 0.0
    top_reward = float(np.mean(rewards[-100:])) if rewards else 0.0
    if st.session_state.trained_algorithm == "Q-Learning":
        top_path = get_path(st.session_state.q_table)
        top_time = st.session_state.get("last_training_time", None)
    else:
        top_path = get_dqn_path(st.session_state.dqn_agent)
        top_time = st.session_state.get("last_training_time", None)
    top_steps = max(0, len(top_path) - 1)
else:
    top_success, top_reward, top_time, top_steps = 0.0, 0.0, None, 0

m1, m2, m3, m4 = st.columns(4, gap="medium")
with m1:
    st.metric("Success rate", f"{top_success:.1f}%" if st.session_state.trained_algorithm else "—", "last 100 episodes" if st.session_state.trained_algorithm else "Train an agent first")
with m2:
    st.metric("Average reward", f"{top_reward:.2f}" if st.session_state.trained_algorithm else "—", "last 100 episodes" if st.session_state.trained_algorithm else "Reward trend")
with m3:
    st.metric("Training time", f"{top_time:.2f} sec" if top_time is not None else "—", st.session_state.trained_algorithm or "Per training run")
with m4:
    st.metric("Route length", f"{top_steps} steps" if st.session_state.trained_algorithm else "—", "learned path" if st.session_state.trained_algorithm else "Shortest safe route")

st.divider()

# Training analytics and maze are shown together so the maze fits in the viewport.
analytics_col, maze_col = st.columns([1.12, 0.95], gap="medium")
with analytics_col:
    st.subheader("📈 Training analytics")
    if st.session_state.trained_algorithm is None:
        st.info("Choose an algorithm and click **Train robot** to populate the charts.")
        st.caption("Reward and success-rate trends will appear here after training.")
    else:
        rewards = st.session_state.rewards
        successes = st.session_state.successes
        episodes_axis = np.arange(1, len(rewards) + 1)
        window = min(50, len(rewards))
        smooth_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
        reward_x = episodes_axis[window - 1:]
        smooth_success = np.convolve(successes, np.ones(window) / window, mode="valid") * 100
        tab_reward, tab_success, tab_loss = st.tabs(["Rewards", "Success rate", "Loss (DQN)"])
        with tab_reward:
            st.line_chart({"Episode": reward_x, "Smoothed reward": smooth_rewards}, x="Episode", y="Smoothed reward", height=230, color="#ff7868")
            st.caption("50-episode moving average. Higher rewards generally indicate better progress.")
        with tab_success:
            st.line_chart({"Episode": reward_x, "Success rate (%)": smooth_success}, x="Episode", y="Success rate (%)", height=230, color="#25d6a2")
            st.caption("Moving success rate shows how consistently the agent reaches the goal.")
        with tab_loss:
            losses = st.session_state.dqn_losses
            if st.session_state.trained_algorithm == "Deep Q-Network (DQN)" and losses:
                st.line_chart({"Loss": losses}, height=230, color="#8b83ff")
                st.caption("DQN loss tracks the Q-value learning objective; lower loss alone does not guarantee a better route.")
            else:
                st.info("Train the DQN algorithm to view its loss curve.")

with maze_col:
    st.subheader("🧩 Maze environment")
    st.caption("S = start · G = goal · Yellow = learned route · Navy = wall")
    if st.session_state.trained_algorithm is None:
        path = []
        st.info("Your learned route will appear here after training.")
    elif st.session_state.trained_algorithm == "Q-Learning":
        path = get_path(st.session_state.q_table)
    else:
        path = get_dqn_path(st.session_state.dqn_agent)
    st.pyplot(draw_maze(path, compact=True), use_container_width=True)
    if st.session_state.trained_algorithm is not None:
        reached_goal = bool(path and path[-1] == GOAL)
        state_col1, state_col2 = st.columns(2)
        with state_col1:
            st.metric("Route steps", max(0, len(path) - 1))
        with state_col2:
            st.metric("Goal reached", "Yes" if reached_goal else "Not yet")
        if not reached_goal:
            st.warning("Try more episodes or adjust exploration to help the agent find a complete route.")

# Comparison area: side-by-side cards, with each metric on the same scale where possible.
st.divider()
with st.expander("⚖️  ALGORITHM ARENA  |  Compare Q-Learning vs DQN", expanded=st.session_state.comparison_results is not None):
    st.write("Train both algorithms on the same maze and compare their performance.")
    comparison_episodes = st.select_slider("Episodes per algorithm", options=[100, 250, 500, 1000, 1500], value=500, key="comparison_episodes")
    if st.button("🚀 Compare both algorithms", type="primary", key="compare_algorithms_button"):
        with st.spinner("Training both algorithms... DQN may take a minute."):
            comparison_results = compare_algorithms(
                episodes=comparison_episodes,
                alpha=alpha if algorithm == "Q-Learning" else 0.15,
                gamma=gamma,
                epsilon=epsilon,
            )
        st.session_state.comparison_results = comparison_results
        st.success("Both algorithms have finished training!")
        st.rerun()

    if st.session_state.comparison_results is not None:
        results = st.session_state.comparison_results
        q_result, dqn_result = results["Q-Learning"], results["DQN"]
        q_success = 100 * float(np.mean(q_result["successes"][-100:]))
        dqn_success = 100 * float(np.mean(dqn_result["successes"][-100:]))
        q_reward = float(np.mean(q_result["rewards"][-100:]))
        dqn_reward = float(np.mean(dqn_result["rewards"][-100:]))
        q_route = max(0, len(q_result["path"]) - 1)
        dqn_route = max(0, len(dqn_result["path"]) - 1)
        q_card, dqn_card, chart_card = st.columns([1, 1, 1.25], gap="medium")
        with q_card:
            st.markdown("### 🟠 Q-Learning")
            st.success("Goal reached" if q_result.get("goal_reached", q_result["path"][-1] == GOAL) else "Route incomplete")
            st.metric("Success rate", f"{q_success:.1f}%")
            st.metric("Average reward", f"{q_reward:.2f}")
            st.metric("Training time", f"{q_result['training_time']:.2f} sec")
            st.metric("Route length", f"{q_route} steps")
        with dqn_card:
            st.markdown("### 🟣 DQN")
            st.success("Goal reached" if dqn_result.get("goal_reached", dqn_result["path"][-1] == GOAL) else "Route incomplete")
            st.metric("Success rate", f"{dqn_success:.1f}%")
            st.metric("Average reward", f"{dqn_reward:.2f}")
            st.metric("Training time", f"{dqn_result['training_time']:.2f} sec")
            st.metric("Route length", f"{dqn_route} steps")
        with chart_card:
            st.markdown("### Performance comparison")
            st.caption("Separate metrics are shown to avoid mixing percentages, rewards, and seconds on one scale.")
            st.bar_chart({"Algorithm": ["Q-Learning", "DQN"], "Success rate (%)": [q_success, dqn_success]}, x="Algorithm", y="Success rate (%)", height=190, color="#25d6a2")
            st.bar_chart({"Algorithm": ["Q-Learning", "DQN"], "Average reward": [q_reward, dqn_reward]}, x="Algorithm", y="Average reward", height=190, color="#ff7868")
            st.bar_chart({"Algorithm": ["Q-Learning", "DQN"], "Training time (sec)": [q_result["training_time"], dqn_result["training_time"]]}, x="Algorithm", y="Training time (sec)", height=190, color="#8b83ff")
        st.markdown("### Learned routes")
        route_col1, route_col2 = st.columns(2, gap="medium")
        with route_col1:
            st.markdown("**Q-Learning route**")
            st.pyplot(draw_maze(q_result["path"], compact=True), use_container_width=True)
        with route_col2:
            st.markdown("**DQN route**")
            st.pyplot(draw_maze(dqn_result["path"], compact=True), use_container_width=True)
        if abs(q_success - dqn_success) < 1e-9:
            st.info("Both algorithms achieved the same success rate in this experiment. Compare reward and training time as well.")
        elif q_success > dqn_success:
            st.info("Q-Learning achieved the higher success rate in this experiment.")
        else:
            st.info("DQN achieved the higher success rate in this experiment.")

st.caption("MazeMind RL · A small reinforcement-learning lab for exploring agent behavior.")
