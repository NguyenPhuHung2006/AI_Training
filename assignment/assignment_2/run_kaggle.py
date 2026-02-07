from kaggle_environments import make
from submission import act

env = make("connectx", debug=True)
env.run(["random", act])
env.render(mode="human")
