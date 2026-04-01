# distributed-self-play

scalable self-play training across multiple machines. the infrastructure behind training game-playing agents at scale.

## what this is

- **actor-learner architecture** — actors generate games in parallel, learner trains the neural network on the stream of data
- **redis communication** — actors push experience to a shared replay buffer via redis. learner pulls batches asynchronously
- **model versioning** — actors periodically pull the latest model weights. play against recent versions of yourself, not just the current one (avoids forgetting)
- **ELO tracking** — monitor training progress via estimated ELO ratings against a fixed set of benchmark agents

## running it

```bash
pip install -r requirements.txt
redis-server &
python learner.py &
python actor.py --num-workers 8
```

## why distribute

self-play is embarrassingly parallel on the data generation side: each game is independent. the bottleneck is the learner's GPU. by running many actors on CPUs (or cheap machines) feeding a single GPU learner, you can generate training data 10-100x faster than single-machine training. this is basically the alphazero infrastructure pattern, simplified.
