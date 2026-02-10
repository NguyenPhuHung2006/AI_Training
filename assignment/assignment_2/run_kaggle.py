from kaggle_environments import make
from neural_network.submission import act

env = make("connectx", debug=True)
env.run(["random", act])
env.render(mode="human")
