---
date: 2026-03-19
tags: [synthesis, ai, engineering, mental-models, career, learning]
---

# Understanding AI From First Principles: A Framework for Engineers Who Build

*Synthesized from a conversation about what separates engineers who use AI tools from engineers who understand them — March 2026*

---

## The Core Thesis

Most engineers today are vibe-coding with AI tools and consuming podcasts. They operate one layer above the abstraction. The highest-leverage move is to go below it: read the source papers, understand the mechanics, and build small tools with tight feedback loops. This is how you develop **taste** — which is just a compressed value function trained on many episodes of building, shipping, evaluating, and iterating.

The path isn't strictly top-down. Many engineers start with tools, build aggressively, and only dive into first principles once they hit repeated failure modes they can't debug from the surface. The key isn't where you start — it's whether you eventually go down the stack when abstractions break.

## The Compiler Analogy

Not understanding LLMs, gradient descent, loss functions, and RL as an engineer in 2026 is like not understanding compilers as a programmer in the 1990s. You could write code without knowing how compilers work. It ran. But when it broke, you had no mental model for *why*. You couldn't reason about what the machine was actually doing versus what your source code said it was doing.

Your peers are writing prompts the same way. It works until it doesn't, and when it breaks, they can't distinguish between a context window problem, a reward hacking artifact, a tokenization issue, or a genuine capability gap. So they retry and hope. That's jiggling code until the compiler stops complaining.

But the deeper failure mode isn't debugging — it's overbuilding. AI coding tools are so fluent at generating correct code that the harder question shifts from "how do I build this?" to "should this code exist at all?" A team can spend five sprints producing a working service — factories, clients, auth providers, tests, all clean and passing — when the actual job was composing three existing CLI tools with a shell script. The AI was excellent at the *how*. What was missing was judgment about the *whether*.

This creates three levels of understanding, not two. Using the tool. Reasoning about the tool. And reasoning about what the tool should and shouldn't be pointed at.

When you understand attention, you write better prompts because you know what the model can "see." When you understand RLHF, you understand why models are sometimes confidently wrong. When you understand MoE, you get why queries route to different capability levels. These gaps widen over time, not narrow.

## The Reading List

**Foundational architecture — understand what you're building on:**

- **Attention Is All You Need** (Vaswani et al., 2017) — the original Transformer
- **Flash Attention** (Dao et al., 2022) — hardware-aware algorithm design; connects infra engineering to inference economics
- **Chinchilla** (Hoffmann et al., 2022) — compute-optimal scaling; changed how everyone thinks about training budgets
- **DeepSeek MoE** — mixture of experts; explains heterogeneous routing and why different queries hit different capability levels

**Agentic architectures — the paradigm you're working in:**

- **ReAct** (Yao et al., 2023) — reasoning + acting; the pattern under every agentic loop including Claude Code and OpenClaw
- **Toolformer** (Schick et al., 2023) — LLMs learning to call tools autonomously
- **Voyager** (Wang et al., 2023) — open-ended agent with skill libraries; the "skills" concept in OpenClaw traces here

**RL and alignment — understand the reward signal:**

- **InstructGPT / RLHF** (Ouyang et al., 2022) — the bridge between "RL as math" and "RL as taste"; human preference *is* the reward signal
- **Reward Is Enough** (Silver et al., 2021) — reward maximization alone produces intelligent behavior
- **Sutton & Barto's RL book** (chapters 1–6) — states, actions, rewards, value functions, exploration vs. exploitation

**Recommendation systems — the domain layer:**

- **Actions Speak Louder Than Words** (Meta) — behavioral signal over stated preference
- **DLRM** (Naumov et al., 2019) — Meta's deep learning recommendation architecture; the reference for this paradigm

**The meta-lesson:**

- **The Bitter Lesson** (Sutton, 2019) — general methods leveraging computation always beat hand-engineered cleverness; challenges your instincts in a productive way

## Munger's Latticework, Applied to Neural Networks

Charlie Munger's "mental models latticework" — building understanding across disciplines piece by piece — maps almost perfectly onto neural network concepts. He just never made the connection.

| Munger Concept | Neural Network Equivalent |
|---|---|
| "Show me the incentive, I'll show you the outcome" | Loss functions — what you optimize for is what you get, including side effects |
| Incremental improvement through feedback | Gradient descent — feel the slope, take a step, feel again |
| Circle of competence | Attention mechanisms — learning what to attend to given context |
| Distrust of backtested strategies | Overfitting — fitting noise in historical data |
| Latticework of mental models | Representation learning — hidden layers building transferable features |
| Humility about prediction | Regularization — penalizing overconfidence |

The latticework itself is a description of what hidden layers do: building intermediate representations from raw experience that transfer across domains.

## Building Taste Through RL Reps

What developing taste looks like in practice is indistinguishable from reinforcement learning:

- **You are the policy.** Each project is an episode. The "reward" is whether the tool actually gets used or rots in a repo.
- **Your taste is the learned value function.** It tells you what's worth building next before you build it.
- **Each tool you ship is a training step.** Market Monitor, ATR backtester, cashflow system, Obsidian synthesis — these aren't side projects. They're training data for your internal model.
- **The tight feedback loop is everything.** If your email parser doesn't work, you know immediately. No faking it. This is what separates building from consuming. (A caveat: this is true for deterministic systems. LLM outputs are probabilistic — failure is a silent hallucination or a subtle drop in recall, not a loud crash. Building taste with AI systems means learning to design evals, not just eyeballing outputs. Without that, your "tight feedback loop" is just vibes.)
- **The negative reward matters too.** A tool that gets used but costs 10x more to maintain than the alternative is a reward hacking artifact — the metric went up but the real objective wasn't optimized. The reps that develop taste include the reps where you *didn't* build something — where you recognized the pieces were already there and the job was glue, not architecture.

The consumption loop (podcast → hype → vibe code → post → move on) produces wide, shallow knowledge. It's the equivalent of reading macro Twitter instead of building a position sizing model. The building loop (read source material → build tool → evaluate → iterate) produces deep, compounding understanding.

## The Composition Trap

AI coding tools make building from scratch feel cheaper than understanding what already exists. Reading docs for a CLI tool, learning pipeline syntax, and mapping existing config files is slow, unglamorous work with no visible output. Generating a service with factories and tests feels productive immediately. The agent fills your screen with green diffs and passing tests. It looks like progress.

But AI drove the cost of *writing* code to near zero while the cost of *maintaining*, reading, and debugging code remains tied to human cognitive limits. Every generated line is maintenance debt signed in someone else's name. Composing existing tools doesn't just feel more elegant — it minimizes your surface area of liability. The team that glues three battle-tested CLIs together owns a shell script. The team that generated a custom service owns a deployment, a runtime, secrets, monitoring, and a migration path when upstream libraries deprecate.

AI collapses the cost of execution. The bottleneck shifts to judgment: what to build, how to evaluate it, and when not to build at all. Engineers who don't develop that judgment will overbuild. Engineers who do will compound leverage.

The "Bitter Lesson" of AI-assisted engineering isn't Sutton's original point — that massive compute beats human cleverness. It's the inverse: relying on the model's raw generative power to write a 1,000-line custom service will eventually lose to the engineer who uses the model to find the exact 3-line command connecting two existing, battle-tested binaries. The hard part is developing the taste to see that *before* you've already built the clever thing.

## The Takeaway

The papers give you better priors. The building gives you reps. Together, they train a value function — taste — that compounds over every project, every domain, every paradigm shift. Your peers are on a flat learning curve. You're on a log curve that's just getting started.

---

*"Every job has a purpose and a task. The task changes. The purpose endures." — Jensen Huang, GTC 2026*

*"Reward is enough." — David Silver et al., 2021*
