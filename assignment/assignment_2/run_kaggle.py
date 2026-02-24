from kaggle_environments import make
from mcts.submission_bitboard import act, reset_agent
from mcts.submission import act_
import traceback

# multiple game
N_GAMES = 50

wins = 0
losses = 0
draws = 0

for i in range(N_GAMES):
    reset_agent()
    env = make("connectx", debug=False)

    try:
        if i % 2 == 0:
            env.run(["random", act])
            my_index = 1
        else:
            env.run([act, "random"])
            my_index = 0

    except Exception:
        traceback.print_exc()
        continue

    result = env.state[my_index].reward

    if result == 1:
        wins += 1
        outcome = "WIN"
    elif result == 0:
        losses += 1
        outcome = "LOSS"
    else:
        draws += 1
        outcome = "DRAW"

    print(f"Game {i+1}: {outcome} | "
          f"Total -> W:{wins} L:{losses} D:{draws}")

print("\nFinal Results")
print("Games:", N_GAMES)
print("Wins:", wins)
print("Losses:", losses)
print("Draws:", draws)
print("Win rate:", wins / N_GAMES)

# single game
# env = make("connectx", debug=True)

# try:
#     env.run(["random", act])
# except Exception:
#     traceback.print_exc()

# env.render(mode="human")

# # ---- print result ----
# for i, agent in enumerate(env.state):
#     print(f"Player {i}:")
#     print("  status:", agent.status)
#     print("  reward:", agent.reward)

# debug
# env = make("connectx", debug=True)
# env.reset()

# obs = env.state[0].observation
# config = env.configuration

# act(obs, config)  
