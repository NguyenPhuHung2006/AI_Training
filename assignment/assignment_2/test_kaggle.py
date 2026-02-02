from kaggle_environments import make

def main():
    print("Creating Kaggle environment...")
    env = make("connectx", debug=True)

    print("Environment name:", env.name)

    print("Running a quick game (random vs random)...")
    env.run(["random", "random"])

    print("Rendering result:")
    print(env.render(mode="ansi"))

    print("\nSUCCESS: kaggle-environments works on this machine.")

if __name__ == "__main__":
    main()
