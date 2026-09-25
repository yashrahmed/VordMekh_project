### KRR

Learning plan for building verifiable world models for LLM agents: explicit
models of state and actions that a solver can check an agent's proposed actions
and plans against.

ASP exercises live in [`logic-programming`](../logic-programming/), planner
experiments in [`classical-planning`](../classical-planning/). For general
background, see the [Handbook of knowledge representation](https://openresearch-repository.anu.edu.au/server/api/core/bitstreams/a75cd087-878c-44ec-999c-0a0f6499f596/content).

#### ASP

##### Foundations

- [ ] Read Vladimir Lifschitz's [Answer Set Programming](https://www.cs.utexas.edu/~vl/teaching/378/ASP.pdf).
  - [x] Chapter 2 for the input language of Clingo.
  - [x] Chapter 3 for combinatorial search.
  - [ ] Chapter 6 for counting, optimization, and symbolic functions.
  - [ ] Chapters 4-5 for the theory of stable models. Skip initially if they slow down practical progress.
- [ ] Work through Potassco's [Answer Set Solving in Practice](https://teaching.potassco.org/) course:
  - [ ] [Introduction](https://teaching.potassco.org/introduction/) for stable models, variables, and safety.
  - [ ] [Modeling](https://teaching.potassco.org/modeling/) for workflow, choices, constraints, and case studies.
  - [ ] [Grounding](https://teaching.potassco.org/grounding/) for rule instantiation.
  - [ ] [Encoding](https://teaching.potassco.org/encoding/) for alternative encodings and performance.
- Use the [Clingo Input Language](https://potassco.org/guide/language/) guide as a syntax reference rather than reading it sequentially.

##### Actions and planning

- [ ] Lifschitz Chapter 7: transition diagrams, effects of actions, inertia, prediction, planning, and concurrency.
- [ ] Study the [Towers of Hanoi Quickstart](https://potassco.org/guide/quickstart/) line by line as a complete Generate-Define-Test-Display planning example.
- [ ] Model a small domain with fluents, actions, preconditions, effects, and inertia, and use it for both prediction and planning.

##### Clingo in an agent loop

- [ ] Potassco course: [Controlling](https://teaching.potassco.org/controlling/) and [Multi-shot solving](https://teaching.potassco.org/msolving/).
- [ ] Learn the [Clingo Python API](https://potassco.org/clingo/python-api/current/): `#program`, `#external`, and incremental grounding.
- [ ] Build a loop in which an LLM proposes an action, Clingo checks its preconditions and computes the next state, and a rejected action returns the violated constraint to the LLM.
- [ ] Explore [telingo](https://github.com/potassco/telingo) for temporal constraints.

##### References

- Gebser, Kaminski, Kaufmann, and Schaub, *Answer Set Solving in Practice*. The book behind the Potassco course.
- Gelfond and Kahl, *Knowledge Representation, Reasoning, and the Design of Intelligent Agents*. Action languages and an ASP-based agent loop with diagnosis and replanning.

#### Classical planning

- [ ] Learn the [Unified Planning](https://unified-planning.readthedocs.io/) framework.

#### TLA+

TLA+ checks properties of a model: invariants that must hold in every
reachable state (safety) and things that must eventually happen (liveness). It
suits concurrent and multi-agent systems.

- [ ] Watch Leslie Lamport's [TLA+ Video Course](https://lamport.azurewebsites.net/video/videos.html) for the core ideas: states, actions as relations between states, invariants, and model checking with TLC.
- [ ] Work through Hillel Wayne's [Learn TLA+](https://learntla.com/) for a practical introduction, including PlusCal, the algorithm-like language that translates to TLA+.
- [ ] Install the [VS Code extension](https://github.com/tlaplus/vscode-tlaplus) to write specs and run the TLC model checker.
- [ ] Encode priests and cannibals in TLA+. Check that the safety invariant holds in every reachable state, and find a solution by asserting that the goal is unreachable and reading the counterexample trace.
- [ ] Try [Apalache](https://apalache-mc.org/), a symbolic model checker for TLA+, on a domain too large for TLC's explicit state enumeration.
- Use the [TLA+ Examples](https://github.com/tlaplus/Examples) repository for worked specs, the [TLA+ summary](https://lamport.azurewebsites.net/tla/summary-standalone.pdf) for syntax, and Lamport's [Specifying Systems](https://lamport.azurewebsites.net/tla/book.html) as the full reference.

#### Neural world models

1. Read Ha and Schmidhuber's [World Models](https://worldmodels.github.io/) for perception, learned dynamics, and a controller that uses the learned model.
2. Study JEPA and its variants in this order:
   - [JEPA architecture and motivation](https://openreview.net/forum?id=BZ5a1r-kVsf): prediction in representation space rather than reconstruction of every pixel. Treat this as a position paper, not a demonstrated complete system.
   - [I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/): predicting representations of missing image regions.
   - [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/): extending the idea to video and temporal structure.
   - [V-JEPA 2 and V-JEPA 2-AC](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/): actionless video pretraining and action-conditioned prediction for planning and control.
   - For each model, ask what its representation preserves, what inputs its predictor receives, and how its predictions can support action selection.
3. Study [DreamerV3](https://danijar.com/project/dreamerv3/) for learning a world model and improving behavior using imagined trajectories.
4. Build a small block-pushing experiment using simulation data from one of the physics engines below. Train an action-conditioned predictor, then measure prediction errors and task performance when masses or friction change.

#### Physics engines

Simulators for generating physical interaction data.

- [PyBullet](https://github.com/bulletphysics/bullet3): Python bindings for the Bullet engine, a common choice for robotics simulation.
- [Rapier](https://rapier.rs/): a 2D and 3D physics engine written in Rust.
- [Blender](https://www.blender.org/): 3D modeling and rendering with built-in rigid-body physics, useful for producing visual observations.
