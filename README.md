# mujoco-maze

Some maze environments for reinforcement learning(RL) using [mujoco-py] and
[openai gym][gym].

Thankfully, this project is based on the code from  [rllab] and [tensorflow/models][models].

## Implemeted Environments

- Distance based rewards
  - AntMaze-v0
  - AntPush-v0
  - AntFall-v0
  - PointMaze-v0
  - PointPush-v0
  - PointFall-v0

- Goal rewards + step penalty
  - AntMaze-v1
  - AntPush-v1
  - AntFall-v1
  - PointMaze-v1
  - PointPush-v1
  - PointFall-v1

## License
This project is licensed under Apache License, Version 2.0
([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

[gym]: https://github.com/openai/gym
[models]: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
[mujoco-py]: https://github.com/openai/mujoco-py
[rllab]: https://github.com/rll/rllab
