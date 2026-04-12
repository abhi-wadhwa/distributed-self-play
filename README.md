# distributed-self-play

actors generate games on CPUs. a single GPU learner trains on the stream. alphazero infrastructure, simplified to the point where it runs on hardware i can afford.

self-play is embarrassingly parallel on the data side. the bottleneck is always the GPU. always.
