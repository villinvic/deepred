# Setup
1 - clone this repo into the folder containing your **activated** venv, and run at the project's root:
```commandline
chmod 777 installs.sh
./installs.sh
pip install -e .
```
To test the install, run
```commandline
python3 tests/env_test.py
```
This should open a controllable gameboy window,  (through the commandline, c.f. deepred/polaris_env/action_space.py)


# TODOs

- rewards
- full observation space
- model
- update metrics
- early stopping if no rewards for a while
- Check TODOs

## Can add:
- use of items outside of battle (could be automated)
- use of TMs/HMs on pokemons.