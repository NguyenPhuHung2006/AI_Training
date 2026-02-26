from kaggle_environments import make
from mcts.submission_bitboard import act, reset_agent
from mcts.submission import act_numpy, reset_agent_numpy
import traceback

# multiple game
N_GAMES = 50

wins = 0
losses = 0
draws = 0
errors = 0

for i in range(N_GAMES):
    reset_agent()
    reset_agent_numpy()
    env = make("connectx", debug=True)

    try:
        if i % 2 == 0:
            env.run([act_numpy, act])
            my_index = 1
        else:
            env.run([act, act_numpy])
            my_index = 0

    except Exception:
        traceback.print_exc()
        continue

    r0 = env.state[0].reward
    r1 = env.state[1].reward
        
    if r0 is None or r1 is None:
        errors += 1
        outcome = "E"
        error_player_index = 0 if r0 is None else 1
        error_player_name = "my agent" if error_player_index == my_index else "opponent agent"
        print(f"error from " + error_player_name)
        env.render(mode="human")
    elif r0 == r1:
        draws += 1
        outcome = "D"
        env.render(mode="human")
    elif env.state[my_index].reward == 1:
        wins += 1
        outcome = "W"
    else:
        losses += 1
        outcome = "L"
        env.render(mode="human")

    print(f"Game {i+1}: {outcome} | "
          f"Total -> W:{wins} L:{losses} D:{draws} E:{errors}")

print("\nFinal Results")
print("Games:", N_GAMES)
print("Wins:", wins)
print("Losses:", losses)
print("Draws:", draws)
print("Errors:", errors)
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
