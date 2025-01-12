 # Implementation of RedArena

This document outlines the steps to implement **RedArena** based on the existing framework provided by classes like `PolarisRed`, `PolarisRedRewardFunction`, and others. Follow the objectives below in order.

---

## Objectives

### 1. **Implement `PolarisRedArenaRewardFunction`**

Define and complete the reward function for the RedArena environment. You have full freedom to redefine rewards as you see fit. The goal is to design rewards that encourage policies with the following behaviors:
- **Win trainer battles.**
- **Level up lower-leveled Pokémon** to promote balanced team growth.
- **Interact optimally with wild Pokémon** (catch, defeat, or flee based on your chosen criteria).

#### Additional Notes:
- The episode should terminate if a battle is stalling. Count such cases as a loss.

---

### 2. **Finish Implementing `BattleSampler`**

Complete the `BattleSampler` class to handle random battle generation. This class should:
- **Sample a random battle**, including:
  - Bag content.
  - Battle nature (wild or trainer). Load either `wild_battle.save` or `trainer_battle.save`.
  - Team composition:
    - Random party count (e.g., 1 for wild Pokémon battles).
    - Random Pokémon, levels, experience, learnable moves, EVs, and IVs.
  
#### Known Issue:
- The method `inject_team_to_ram` is not functioning correctly.

#### Steps to Fix:
1. Run `tests/red_arena_human.py` with:
   - `wild_battle_chance=1`.
   - `team = team[:1]` (one Pokémon per team, see `battle_sampler.py`, line 487).
2. Print RAM values of interest in the hook (`polaris_red_arena.py`, line 154) to verify if the injected values are correct.
3. Update `inject_team_to_ram` until it works as expected. Currently, it works if the party count is set to 1 and no changes are made to the opponent’s team.
4. Repeat steps 1–3 with:
   - `wild_battle_chance=0`.
   - Larger teams, e.g., `team = team[:2]`, etc.

#### Additional Notes:
- You might have to change the `wild_battle.save` / `trainer_battle.save` files to properly start the episode in a battle
  or even to make `inject_team_to_ram` work.

---

### 3. **Implement `RedArena`**

The `RedArena` class governs the environment’s behavior. Implement the following methods:

- **`reset`:**
  - Reset the environment to its initial state, simulating the start of a new episode.

- **`step`:**
  - Process an action in the environment and return:
    - The next observation.
    - The corresponding reward.
    - A flag indicating whether the episode has terminated.
  - **Behavior:**
    1. At the episode's start, the battle begins, and the agent can prepare its team using the "roll_action." However, the agent cannot walk away (battle is mandatory).
    2. Terminate the episode if:
       - The maximum number of actions (`episode_length`) is exceeded.
       - The agent loses, wins, or catches the opposing Pokémon.

- **`get_episode_metrics`:**
  - Pass relevant metrics (e.g., `winrate`) as a dictionary to `wandb`.

---

### 4. **Additional Improvements**

Review the `PolarisRedArenaObservationSpace` class to identify areas for improvement or additional features.

---

## Testing

To test your implementation, run:

```bash
python3 tests/red_arena_human.py