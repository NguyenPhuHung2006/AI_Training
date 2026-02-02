from kaggle_environments import make
from agent import agent
from submission import act

env = make("connectx", debug=True)
env.run(["random", agent])
env.render(mode="human")
