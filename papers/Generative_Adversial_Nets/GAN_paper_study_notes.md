# GAN Paper — Study Notes

My working notes on **Generative Adversarial Nets** (Goodfellow et al., 2014), built up
from going through the paper piece by piece. Written intuition-first, with the parts I kept
getting stuck on spelled out the long way.

---

## Contents

1. [The one-sentence idea](#1-the-one-sentence-idea)
2. [Architecture of G and D (what the paper actually says)](#2-architecture-of-g-and-d)
3. [Can an MLP handle images?](#3-can-an-mlp-handle-images)
4. [The value function, term by term](#4-the-value-function-term-by-term)
5. [⭐ The min/max confusion (my main sticking point)](#5--the-minmax-confusion)
6. [What is the `log` even doing?](#6-what-is-the-log-even-doing)
7. [The non-saturating trick](#7-the-non-saturating-trick)
8. [Why the game has a clean optimum (optimal D → JSD)](#8-why-the-game-has-a-clean-optimum)
9. [How it's actually trained](#9-how-its-actually-trained)
10. [The experiment I ran](#10-the-experiment-i-ran)
11. [One-line summaries to remember](#11-one-line-summaries)

---

## 1. The one-sentence idea

Train two networks against each other:

- **Generator G** — makes fake data that should look real.
- **Discriminator D** — tells real data from fake.

The paper's metaphor: **G is a counterfeiter** making fake currency, **D is the police**
catching it. Each forces the other to improve until the fakes are indistinguishable from real.

---

## 2. Architecture of G and D

The paper **deliberately under-specifies this** — the contribution is the *framework*, not a
specific network. What it actually states:

- Both **G and D are multilayer perceptrons (MLPs)**.
- `G(z; θ_g)` maps a noise vector `z` → a sample in data space.
- `D(x; θ_d)` maps an input → a **single scalar** in `[0,1]` = "probability this is real."
- **Generator** used a mix of **ReLU + sigmoid** activations.
- **Discriminator** used **maxout** activations + **dropout**.
- Noise `z` enters **only at the bottom layer** of G (theory allows noise at every layer, but
  they didn't).
- One variant: a **convolutional discriminator + "deconvolutional" generator** for CIFAR-10.

**Not given in the paper:** layer counts, widths, latent dimension, learning rates. Those live
in the released code. The real architecture recipe that made GANs work came a year later with
**DCGAN (2015)** — strided convolutions, batch norm, no fully-connected hidden layers.

> Takeaway: this paper is the *idea*. The architecture I actually use (SwinIR + PatchGAN, etc.)
> is several generations of "give the network back its sense of space" later.

---

## 3. Can an MLP handle images?

Yes — and the original GAN did. **But "can" and "should" diverge.**

**How:** flatten the image into a vector. A 28×28 MNIST digit → a 784-dim vector. D takes 784
numbers in. G outputs 784 numbers → reshape back to 28×28.

**Why it's bad:**

- **Flattening destroys spatial structure.** The MLP has no built-in idea that two pixels are
  neighbours — every pixel is an independent coordinate.
- **No weight sharing.** An edge detector learned in one corner is useless elsewhere. A
  convolution slides the *same* filter everywhere (translation equivariance) — learn once, apply
  everywhere.
- **Parameter explosion.** First dense layer connects every pixel to every hidden unit. Fine at
  28×28; at 256×256×3 ≈ 196k inputs it's tens of millions of weights in one layer.
- **Quality ceiling.** MLP-GAN samples are small, low-res, blurry.

This is exactly the gap **DCGAN** closed by going convolutional. The MLP version is a
proof-of-concept that the *adversarial objective* works, not a serious image generator.

---

## 4. The value function, term by term

$$\min_G \max_D V(D,G) = \underbrace{\mathbb{E}_{x \sim p_{data}}[\log D(x)]}_{\text{term 1}} + \underbrace{\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]}_{\text{term 2}}$$

- `D(x)` = a scalar in `[0,1]`. D's "probability this is real." Same for `D(G(z))` on a fake.
- **Term 1** `E_x[log D(x)]` — only sees **real data** (`x ~ p_data`). D maximizes → push
  `D(x) → 1` ("say real on real"). **G is not in this term** — it can't touch real data.
- **Term 2** `E_z[log(1 − D(G(z)))]` — only sees **fakes** (`z` is noise → `G(z)` is a fake). D
  maximizes → push `D(G(z)) → 0` ("say fake on fake").
- **G appears only in term 2**, and `min_G` means G wants the value **small**, which means
  pushing `D(G(z)) → 1` (fool D).

So `min_G max_D` tells you who optimizes which way; term 1 is D's solo lesson on real data;
term 2 is the **battleground** where G and D fight over the number `D(G(z))`.

---

## 5. ⭐ The min/max confusion

**This is the part I kept getting caught on.** The trap:

> "maximize" / "minimize" refer to the **whole value V** — NOT to `D(G(z))` directly.

When I read "D maximizes," my brain wanted to apply it to the nearest number `D(G(z))`. Wrong.
They want the **total V** to go their way; `D(G(z))` is just a knob they turn.

### The fix: always go through V in two steps

1. **Find the player.** `max_D` or `min_G`?
2. **That sets the goal for the whole V:** D → big, G → small.
3. **Then** ask: which way do I turn my knob `D(G(z))` to push V that way? → plug in numbers.

Never reason directly about whether `D(G(z))` "should" be big or small. Always go through V first.
That extra hop is the thing I was skipping.

### Plug-in-the-numbers (do this every time it feels slippery)

**Term 2 = `log(1 − D(G(z)))`, from D's side (D wants V big):**

| `D(G(z))` | `1 − D(G(z))` | `log(...)` | meaning |
|---|---|---|---|
| 0.99 (D fooled) | 0.01 | **−4.6** | D's worst |
| 0.50 | 0.50 | −0.7 | |
| 0.01 (D catches it) | 0.99 | **−0.01** | D's best |

Read down: `−4.6 → −0.7 → −0.01` — value **climbs** as `D(G(z))` **falls**. So **D makes V big by
pushing `D(G(z))` down.** The `1 − D` inside is what flips the direction.

**Term 1 = `log D(x)`, from D's side (no `1 − D`, so no flip):**

| `D(x)` | `log D(x)` |
|---|---|
| 0.99 (correct) | −0.01 |
| 0.50 | −0.69 |
| 0.01 (wrong) | −4.6 |

D makes V big by pushing `D(x)` **up**. Rewarded for calling real data real.

### The insight that finally unstuck me: one number, two readings

The same `D(G(z))` has **two bosses pulling opposite ways**:

- **D** turns it **down** (toward 0) to make V **big** → "fake is fake."
- **G** turns it **up** (toward 1) to make V **small** → "fool you into saying real."

Same knob, two hands. When reading about D, **hold G's hand still**; when reading about G, freeze
D. The dizziness came from tracking both at once.

### "If G fooled D, why does it get MORE penalty?" — it doesn't

`D(G(z)) = 0.99 → log(0.01) = −4.6`. That `−4.6` is **not a penalty for G.** I was reading G's
reward through D's eyes.

- To **D** (wants big): `−4.6` = "I failed." → penalty.
- To **G** (wants small): `−4.6` = "I drove it as low as possible." → **reward / the prize.**

It's a tug-of-war over the *same rope*. Whatever is bad for one player is good for the other
(zero-sum-ish). Negative *looks* bad, but for the **minimizing** player, negative is the goal. I
was just still sitting in D's chair. Slide to G's chair → low is what G wants.

---

## 6. What is the `log` even doing?

The `log` turns a **probability** into a **reward/penalty** with a useful shape: **gentle when
you're right, brutal when you're confidently wrong.**

`log D(x)` (reward for calling real data real):

- `D(x) = 0.9` → `log(0.9) ≈ −0.1` (basically "fine")
- `D(x) = 0.5` → `≈ −0.7`
- `D(x) = 0.01` (confidently wrong) → `≈ −4.6` (**huge**)

As probability → 0, `log → −∞`. So the loss screams loudest exactly when the network is
**confidently wrong** → it races to kill its worst mistakes first. Without `log`, being 99% vs
100% right would look almost the same and confident blunders wouldn't be punished extra.

Second reason it must be `log`: the whole objective is a **log-likelihood**. D is estimating a
probability; summing log-probabilities is how you maximize likelihood (probabilities multiply,
`log` turns ×into +, easier to optimize). The `−log 4` / JSD result falls straight out of these
being logs.

---

## 7. The non-saturating trick

The trick is entirely about the **slope (gradient)**, not the value.

- **Original (saturating):** G minimizes `log(1 − D(G(z)))`.
- **Fix (non-saturating):** G maximizes `log D(G(z))`.

Same goal (push `D(G(z))` up), same fixed point. The difference is **where the curve is steep.**

### Why it matters early in training

Early on G is bad → D catches everything → `D(G(z)) ≈ 0`. Watch a small improvement
`0.01 → 0.02`:

**Saturating** `log(1 − D)`:

- `D(G(z)) = 0.01 → log(0.99) = −0.0101`
- `D(G(z)) = 0.02 → log(0.98) = −0.0202`
- loss moved ≈ **0.01** → almost flat → **tiny gradient → G barely learns**. ("saturation")

**Non-saturating** `log D`:

- `D(G(z)) = 0.01 → log(0.01) = −4.605`
- `D(G(z)) = 0.02 → log(0.02) = −3.912`
- loss moved ≈ **0.69** → ~**70× more** for the *same* improvement → **strong gradient**.

The original `log(1 − D)` is **flat near 0** (its steep part is over near `D(G(z)) = 1`, where G
hasn't reached yet). `log D` is **steep near 0** — right where G actually starts. So just pick the
loss whose steep part sits where G needs help.

> Mechanism in one line: the saturating gradient gets multiplied by `D(G(z)) ≈ 0`; the
> non-saturating one by `1 − D(G(z)) ≈ 1`.

---

## 8. Why the game has a clean optimum

Two moves.

### Move 1 — the best possible D (G frozen)

Ask: at a point `x`, what should D answer? It's just a poll — "of everything that lands here,
what fraction was real?":

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

**The optimal discriminator is a density ratio.** Where real and fake show up equally often,
`D* = ½` (genuinely ambiguous → coin flip).

### Move 2 — make G fight that best D

Plug `D*` back in; the algebra collapses to what G is *really* minimizing:

$$C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{data} \,\|\, p_g)$$

Ignore the constant `−log 4`. **JSD = Jensen–Shannon divergence = "how different are the two
distributions?"** It's `0` iff they're identical, positive otherwise.

So **G minimizing its loss = G making its fake distribution match the real one.** Best case: JSD
→ 0, i.e. **`p_g = p_data`**, and at that point `D* = ½` everywhere — even the perfect
discriminator is reduced to coin-flipping because there's nothing left to tell apart.

**Punchline:** you can't compute "how different are real and fake?" directly (no formula for
`p_data`). Training a classifier to its best **secretly computes that difference for you.** D is a
stand-in for a distance you could never write down.

---

## 9. How it's actually trained

You can't solve D to optimality every step (too slow, overfits). So alternate:

1. For **k steps**: sample real `x` + noise `z`, **ascend** D's gradient (improve D). Paper used
   **k = 1**.
2. Then **one descent step** on G (improve G against the current D).

**Caveat (theory vs practice):** the clean `p_g = p_data` proof assumes D is perfect each step and
is about idealized *distributions* `p_g`. Real GANs only tweak network weights `θ_g` — a limited,
non-convex family — so the guarantee doesn't strictly hold. It just works well in practice.

**Failure mode named in the paper:** the *"Helvetica scenario"* = **mode collapse** — if G is
updated too much without D keeping pace, G collapses many `z` onto one `x` and loses diversity.
Keeping D and G synchronized is the core practical tension.

---

## 10. The experiment I ran

Built a notebook (`gan_saturating_vs_nonsaturating.ipynb`) — MLP G and D on a 2D
8-mode-ring distribution — comparing the two losses:

- **Analytical:** plotted gradient magnitude vs `D(G(z))`. Saturating ∝ `D(G(z))` (→0 early);
  non-saturating ∝ `1 − D(G(z))` (≈1 early).
- **At initialization:** with a confident D (mean `D(G(z)) = 0.0006`), measured the real
  parameter-gradient norm:
  - saturating ‖∇‖ = **0.0047**
  - non-saturating ‖∇‖ = **5.30**
  - → **~1,135× stronger**, from the *identical* G and batch.
- **Full training:** non-saturating gets a usable signal from step 1; saturating crawls early.

That `1,135×` is the same `0.01 vs 0.69` slope gap from §7, multiplied through the whole MLP.

---

## 11. One-line summaries

- **The game:** G fakes, D catches; competition makes both better.
- **min/max:** the words refer to the **whole V**, not to `D(G(z))`. Find the player → set V's
  direction → *then* plug in numbers to see which way the knob moves.
- **One number, two readings:** the same `D(G(z))` is D's loss and G's win. Mind which chair
  you're in.
- **`log`:** turns probability into a likelihood score — gentle when right, brutal when
  confidently wrong.
- **Non-saturating trick:** same goal, but its steep part sits at `D(G(z)) ≈ 0` where G starts →
  strong early gradient.
- **Optimal D = density ratio**; forcing G to beat it = forcing **`p_g → p_data`** (minimizing
  JSD).
- **Training:** alternate k D-steps then 1 G-step; proofs are idealized, practice is messier but
  works.
