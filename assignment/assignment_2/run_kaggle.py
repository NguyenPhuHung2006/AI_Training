from kaggle_environments import make
from mcts.submission import act
import traceback

env = make("connectx", debug=True)

try:
    env.run(["random", act])
except Exception:
    traceback.print_exc()

env.render(mode="human")

# ---- print result ----
for i, agent in enumerate(env.state):
    print(f"Player {i}:")
    print("  status:", agent.status)
    print("  reward:", agent.reward)

# debug
# env = make("connectx", debug=True)
# env.reset()

# obs = env.state[0].observation
# config = env.configuration

# act(obs, config)  
