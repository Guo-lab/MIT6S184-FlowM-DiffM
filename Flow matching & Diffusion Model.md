# Mathematics Behind Flow and Diffusion Models

## 1. Dynamics & Density Evolution — ODE vs SDE

> 这一层回答"粒子怎么走"以及"分布随之怎么变"。ODE → Continuity eq；SDE → Fokker-Planck eq。

--- start-multi-column: FlowMatchingOverview
```column-settings
number of columns: 3
border: off
shadow: on
```

> [!note] Gaussian Quick Reference
>
> **Standard vs General**
> $$\mathcal{N}(0, I_d) \quad \longleftrightarrow \quad \mathcal{N}(\mu, \Sigma)$$
>
> **Reparameterization** — sample any Gaussian from $\mathcal{N}(0, I)$:
> $$z \sim \mathcal{N}(\mu, \Sigma) \iff z = \mu + \Sigma^{1/2}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
> - $σ^2 I$ → 球形分布
> - 一般 Σ → 椭球（ellipsoid）
> - Start with a perfectly spherical, centered at 0, then stretch + rotate it into an **ellipsoid**, and ultimately shift it.
>
> **Linear Transform Stays Gaussian**
> $$x \sim \mathcal{N}(\mu, \Sigma) \implies Ax + b (A\mu+b+A\Sigma^{1/2}\epsilon) \sim \mathcal{N}(A\mu + b,\ A\Sigma A^\top)$$
>
> ---
>
> **Score of a Gaussian**
> $$\nabla_x \log \mathcal{N}(x;\, \mu, \sigma^2 I) = -\frac{x - \mu}{\sigma^2}$$
>
> Score points toward the mean — critical for understanding diffusion.
>
> - 从当前点 $x$ 指向均值 $\mu$，是一个**"拉回中心"的力**；分布越"自信"，拉回越强。
> - score 就是指向碗顶（paraboloid）的箭头场
>
> ---
>
> **Diffusion Noising** (diffusion forward process)
>
> At noise level $t$, the forward process gives: **$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t)\, I\right)$**
> or via reparameterization: (把原图 $x_0$ 和纯噪声 $\epsilon$ 按比例混在一起) $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\; \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
>
> ---
>
> **KL Between Two Gaussians**
> $$D_{\text{KL}}\!\left(\mathcal{N}(\mu_1, \Sigma_1) \,\|\, \mathcal{N}(\mu_2, \Sigma_2)\right) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^\top \Sigma_2^{-1}(\mu_2-\mu_1)\right]$$
> For $\Sigma_1 = \Sigma_2 = I$, collapses to $\frac{1}{2}\|\mu_1 - \mu_2\|^2$.
>
> ---
>
> **KL intuition**: 用 Q 去近似真实分布 P，在 P 的视角下，对数差异的期望是多少？即平均每个样本损失多少信息量。
>
> 积分写成期望：
> $$D_{\text{KL}}(P \,\|\, Q) = \int p(x) \ln\!\left(\frac{p(x)}{q(x)}\right) dx \;=\; \underbrace{\int p \ln p}_{E_P[\ln p] = -H(P)} - \underbrace{\int p \ln q}_{E_P[\ln q]} \;=\; E_{x \sim P}\!\left[\ln\frac{p(x)}{q(x)}\right]$$
>
> **Cross-entropy** $-\!E_P[\ln q]$：用 Q 去编码 P 时每个样本的平均编码成本；**entropy** $-\!E_P[\ln p]$：用真实 P 自身编码的最低成本。KL = 用错分布 Q 多付的代价。

--- end-column ---

> [!abstract] SDE vs ODE — Dynamics
>
> **ODE** *(deterministic)*:
> $$\frac{dx}{dt} = v_t(x)$$
>
> **SDE** *(stochastic)*:
> $$dx = v_t(x)\,dt + \sigma(t)\,d\mathbf{w}$$
>
> $$dw = W_{t+dt} - W_t$$
>
> SDE 就是在 ODE 的基础上多加了一项 $\sigma(t)\,d\mathbf{w}$ — 布朗运动（Brownian motion）带来的**一点点噪声**。扩散系数 $\sigma(t)$ 通常很小，所以 SDE 的动力学和 ODE 几乎一样，只是带了一点随机扰动。
>
> **与 FM 的关系**：FM 用的是 ODE，diffusion model 用的是 SDE。
>
> ODE gives a fixed trajectory that there is no way to change the process, while SDE makes it possible to optimize and search over all options. (leading to more diversity)
>
> ---
>
> **Diffusion Model 起点是 数据 + 加噪过程， 学的是逆转噪声过程**
> Forward (SDE): data $\xrightarrow{\sigma(t)d\mathbf{w}}$ noise $\mathcal{N}(0,I)$
> $$dx = f(x, t) dt + g(t) dw$$
> 
> 
> Reverse (learned): noise $\xrightarrow{\nabla\log p_t\text{ or }v_t}$ data
> $$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar{w}$$
> Score用来抵消掉正向扩散带来的熵增

--- end-column ---

> [!example] Density Evolution — ODE & SDE
>
> **Goal:** Learn vector field $v_t(x)$ that transports $p_0 \to p_1$
>
> **Flow ODE** *(deterministic, no noise)*:
> $$\frac{dx}{dt} = v_t(x), \quad x_0 \sim p_0 = \mathcal{N}(0, I)$$
>
> **Continuity equation** *(consistency condition)*:
> $$\frac{\partial p_t}{\partial t} = - \nabla \cdot (p_t\, v_t) $$
> - 来确保数据分布在被"推"向目标分布的过程中，数据的总数（概率总和 1）不会凭空多出来或少
> - 密度的增加（$\partial p / \partial t$），**必须等于**从四周流进来的净流量： $-\text{divergence}$ （汇聚）of flux
>
>
>Flow Matching 
> **起点是 数据 + latent pair，定义怎么从 A 运到 B，模型学搬运速度**
> ---

> [!info] 基于 Score — Score 为何关键
> **Fokker-Planck equation** *(density evolution for SDE $dx = v_t(x)\,dt + \sigma(t)\,d\mathbf{w}$)*:
> $$\boxed{\frac{\partial p_t(x)}{\partial t} =  -\nabla_x \cdot \big(p_t(x)\, v_t(x)\big) + \frac{1}{2}\sigma^2(t)\,\Delta_x p_t(x)}$$
>
> Compare with **Continuity equation** (ODE $\sigma = 0$):
> $$\frac{\partial p_t(x)}{\partial t} = -\nabla_x \cdot \big(p_t(x)\, v_t(x)\big)$$
>
> The extra $\frac{1}{2}\nabla_x \cdot (\sigma^2 p_t \nabla_x \log p_t) = \frac{1}{2}\sigma^2 \Delta_x p_t$ term = diffusion from Brownian noise. When $\sigma \to 0$, Fokker-Planck → Continuity.
>
> **FP 推导直觉**（Itô's lemma）：$dw \sim \sqrt{dt}$，所以 $(dw)^2 \sim dt$ 该项均匀化自动抹平，耗散
>
> **FP ↔ DSM 的关联**：FP 的扩散项 = $\frac{1}{2}\nabla \cdot (\sigma^2 p_t \nabla \log p_t)$，里面的 $\nabla \log p_t$ **正是 DSM 学的 score**。学会 score → (diffusion term = divergence of score field) FP → 就能推 reverse SDE 或 PF-ODE → 就能采样。
>
> ---
>
> **SDE-ODE Equivalence** — score-augmented drift cancels diffusion:
> $$dx_t = \Big[v_t^{\text{target}}(x_t) + \frac{\sigma_t^2}{2}\nabla \log p_t(x_t)\Big]dt + \sigma_t\,d\mathbf{w}_t$$
> Apply Fokker-Planck:
> $$\begin{aligned} \frac{\partial p_t}{\partial t} &= -\nabla \cdot \Big(p_t \big[v_t^{\text{target}} + \frac{\sigma^2}{2}\nabla \log p_t\big]\Big) + \frac{1}{2}\sigma^2 \Delta p_t \\ &= -\nabla \cdot (p_t v_t^{\text{target}}) - \underbrace{\frac{\sigma^2}{2}\nabla \cdot (p_t \nabla \log p_t) + \frac{1}{2}\sigma^2 \Delta p_t}_{=\,0} \\ &= -\nabla \cdot (p_t v_t^{\text{target}}) \quad\longleftarrow\text{ same as ODE!} \end{aligned}$$
> → 在 drift 里补上 $\frac{\sigma^2}{2}\nabla\log p_t$ 就能抵消扩散项 → SDE 和 ODE 有**完全相同的边际分布** → **PF-ODE 等价性**。


--- end-multi-column

## 2. Training — 怎么学会那个场？& 为什么 Conditional 等价于 Marginal？

> FM 学向量场 $v_t$，Score Matching 学梯度场 $\nabla\log p_t$，两者共享同一个 marginalization trick：conditional target 训练 → 自动收敛到 marginal field。

--- start-multi-column: TrainingOverview
```column-settings
number of columns: 3
border: off
shadow: off
```

> [!info] ① Flow Matching — Marginal VF & Continuity
>
> **Training Target: Marginal Vector Field** (conditional expectation of the conditional VF):
> $$v_t^{\text{target}}(x) = \int \underbrace{v_t^{\text{target}}(x \mid z)}_{\text{Cond VF}} \; \cdot \underbrace{\frac{p_t(x \mid z)\, p_{\text{data}}(z)}{p_t(x)}}_{\text{Bayes posterior: } p_t(z \mid x)} \; dz$$
>
> ---
>
> **Continuity equation** *(density evolution for ODE)*:
> $$\frac{\partial p_t(x)}{\partial t} = -\nabla_x \cdot \big(p_t(x)\, v_t(x)\big)$$
> - 概率总和守恒：密度的增加 = 从四周汇聚来的净流入
>
> **Derivation (marginalization trick)**:
> $$\begin{aligned} \frac{\partial p_t(x)}{\partial t} &\stackrel{①}{=} \frac{\partial}{\partial t} \int p_t(x \mid z)\, p_{\text{data}}(z)\, dz \\ &\stackrel{②}{=} \int \frac{\partial}{\partial t} p_t(x \mid z)\, p_{\text{data}}(z)\, dz \\ &\stackrel{③}{=} \int \big[-\nabla_x \cdot (p_t(x \mid z)\, v_t^{\text{target}}(x \mid z))\big]\, p_{\text{data}}(z)\, dz \\ &\stackrel{④}{=} -\nabla_x \cdot \int p_t(x \mid z)\, v_t^{\text{target}}(x \mid z)\, p_{\text{data}}(z)\, dz \\ &\stackrel{⑤}{=} -\nabla_x \cdot \big(p_t(x)\, v_t^{\text{target}}(x)\big) \end{aligned}$$
> $$\text{③ where }\; \frac{\partial}{\partial t}\, p_t(x \mid z) = -\nabla_x \cdot \big(p_t(x \mid z)\, v_t^{\text{target}}(x \mid z)\big)$$
> ① marginalize, ② derivative in ($p_{\text{data}}(z)$ independent of $t$), ③ conditional continuity eq, ④ pull divergence out, ⑤ definition of marginal VF.
> $$\text{where }\; \nabla \cdot v_t(x) = \sum_{i=1}^d \frac{\partial}{\partial x_i} v_t^{(i)}(x)$$
>
> Continuity equation closed under marginalization.
> 
> ---
>
> **Flow Matching loss** *(intractable directly)*:
> $$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, x_t}\big\|u_\theta(x_t, t) - v_t(x_t)\big\|^2$$
>
> **Conditional Flow Matching** *(tractable surrogate ✓)*:
> $$\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t, z, x_t\sim p_t(\cdot\mid z)}\big\|u_\theta(x_t, t) - v_t(x_t \mid z)\big\|^2$$
>
> **Key insight:** 最小二乘（LSM）拟合目标分布的期望值，利用这个特性喂个案懂全局。
> $\mathcal{L}_\text{FM}$ and $\mathcal{L}_\text{CFM}$ share the same minimizer — **train on the conditional, get the marginal**.
>
> 用 sample-level interpolation 替代 distribution-level evolution
>
> ---
>
> ---
>
> **$u_\theta$ 的目标是什么？** 不是逼近条件向量场 $v_t(x \mid x_1)$，而是逼近 **边际向量场** $v_t(x)$：
> $$v_t(x) = \mathbb{E}_{\substack{z \sim p_1 \\ x \sim p_t(\cdot \mid z)}}\!\Big[v_t(x \mid z) \;\Big|\; x_t = x\;\Big]$$
> - t是自变量，在固定的时刻 $t$，对所有可能的样本 $x_1$ 求平均。
> - 在位置 $x$ 处，边际向量场 = 所有可能通向 $x$ 去往$x_1$的条件路径在该点的向量场的**加权平均**。这个平均值只依赖于 $x$ 和 $t$，**不依赖于任何一个特定的 $x_1$**。
>
> **损失函数的梯度等价**:
> $$\boxed{\nabla_\theta \mathcal{L}_\text{FM}(\theta) = \nabla_\theta \mathcal{L}_\text{CFM}(\theta)}$$
>

--- end-column ---

> ↓ 那么，怎么学到这个 Score？

> [!info] Score Matching — Marginal Score & DSM
>
> **Training Target: Marginal Score** (conditional expectation of the conditional score):
> $$s_t^{\text{target}}(x) = \int \underbrace{\nabla_x \log p_t(x \mid z)}_{\text{Cond Score}} \cdot \underbrace{\frac{p_t(x \mid z)\, p_{\text{data}}(z)}{p_t(x)}}_{\text{Bayes: } p_t(z \mid x)} \; dz$$
>
> **Denoising Score Matching** *(tractable surrogate)*:
> $$\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{t, z, x_t \sim p_t(\cdot \mid z)}\big\| s_\theta(x_t, t) - \nabla_{x_t} \log p_t(x_t \mid z) \big\|^2$$
>
> **Derivation** (same marginalization trick, score version):
> $$\begin{aligned} s_t^{\text{target}}(x) &= \nabla_x \log p_t(x) \\ &\stackrel{①}{=} \frac{\nabla_x p_t(x)}{p_t(x)} \\ &\stackrel{②}{=} \frac{\nabla_x \int p_t(x \mid z)\, p_{\text{data}}(z)\, dz}{p_t(x)} \\ &\stackrel{③}{=} \frac{\int \nabla_x p_t(x \mid z)\, p_{\text{data}}(z)\, dz}{p_t(x)} \\ &\stackrel{④}{=} \int \frac{\nabla_x p_t(x \mid z)}{p_t(x \mid z)} \cdot \frac{p_t(x \mid z)\, p_{\text{data}}(z)}{p_t(x)} \, dz \end{aligned}$$
> ① score definition, ② marginalize, ③ derivative in ($p_{\text{data}}(z)$ independent of $x$), ④ multiply by $\frac{p_t(x \mid z)}{p_t(x \mid z)}$
>
> **What is score?** $\nabla_x \log p_t(x)$ = a vector field pointing toward higher density. For Gaussian: $-\frac{x-\mu}{\sigma^2}$ — a restoring force pulling toward the mean. Score tells you "which direction makes the data more likely."
>
> **Why this derivation matters?** 和 CFM 完全同一个 trick：intractable marginal score $s_t(x)$ = Bayes-weighted average of conditional scores $s_t(x \mid z)$。训练时只需要 conditional target（DSM loss），模型自动收敛到 marginal score。不需要知道 $p_t(x)$ 就可以学到它的 score。

--- end-column ---

> [!info] ③ Guidance — Conditional Generation
>
> **Classifier Guidance** *(conditional generation via Bayes)*:
> $$\hat{v}_t^w(x \mid y) = v_t^{\text{target}}(x) + w\, \alpha_t \nabla_x \log p_t(y \mid x)$$
> Unconditional $v_t^{\text{target}}(x)$ + scaled classifier gradient toward $y$. Weight $w$ controls strength.
>
> ---
>
> **Classifier-Free Guidance**:
> $$\hat{u}_t^w(x \mid y) = (1-w)\, u_t^{\text{target}}(x) + w\, u_t^{\text{target}}(x \mid y)$$
> **Derivation** (2 key steps), using $u_t^{\text{target}} = b_t x + \alpha_t \nabla \log p_t$:
> $$\begin{aligned} \hat{u}_t^w(x \mid y) &= u_t^{\text{target}}(x) + w \alpha_t \underbrace{\nabla \log p_t(y \mid x)}_{\text{Bayes: } \nabla \log p_t(x \mid y) - \nabla \log p_t(x)} \\ &= u_t^{\text{target}}(x) - w u_t^{\text{target}}(x) + w u_t^{\text{target}}(x \mid y) \end{aligned}$$
> **Vector intuition**: $u_t^{\text{target}}(x \mid y) - u_t^{\text{target}}(x)$ is a vector pointing toward the conditional. CFG just moves $w$× along that direction from unconditional → no separate classifier to train or evaluate.
> 
> ---
> 
> **Conditional = Marginal + Correction**
> 解耦
> 
> - **模块化** **无需重训大模型**
> - 组合性（Composability）：可以通过不同的 $y$（比如文本、图像、深度图）来引导同一个 Marginal 模型，而不需要为每一种引导方式去重新进行 Marginal Train
> - 解决“数据稀疏”问题：利用在海量数据上学到的 $p(x)$ 的强泛化能力，再配合一个判别器 $p(y|x)$ 提供的微弱方向，就能在几乎没见过 $y$ 的情况下，依然生成高质量的 $x$
>   
> ---
>   
>   可以把“理解世界长什么样”和“理解指令要求什么”这两个任务分开处理，然后再通过数学手段缝合

--- end-multi-column


## 3. Connection — Latent Variable Models

# VAE , #LatentSpaces
> Reference: [VAE 详解（苏剑林）](https://kexue.fm/archives/7725/comment-page-1)
## VAE Loss

$$\mathcal{L}_{\text{VAE}}(\phi, \theta) = \underbrace{-\mathbb{E}_{x \sim p_{\text{data}}(x),\; z \sim q_\phi(\cdot \mid x)}\!\Big[\log p_\theta(x \mid z)\Big]}_{\mathcal{L}_{\text{VAE-Recon}}(\phi, \theta)} + \beta \cdot \underbrace{\mathbb{E}_{x \sim p_{\text{data}}(x)}\!\Big[D_{\text{KL}}\!\big(q_\phi(\cdot \mid x) \;\big\|\; p_{\text{prior}}\big)\Big]}_{\mathcal{L}_{\text{VAE-Prior}}(\phi)}$$
>
> **β-VAE**：$\beta > 1$ 加强 prior 约束 → 更解耦平滑的 latent space；$\beta < 1$ 放宽约束 → 更重重建质量。$\beta = 1$ 退化为标准 VAE。

| Term     | 含义                                                             | 作用                                                                            |
| -------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Reconst. | 样本从 $p_{\text{data}}$ 来 → encoder 出 $z$ → decoder 重建 $x$ 的对数似然 | 逼迫精准还原                                                                        |
| Prior    | 对数据分布求期望的 KL：后验 $q_\phi$ 偏离先验 $p_{\text{prior}}$ 的程度           | 正则化，强迫模仿正态分布。 防止 $z$ 退化成单纯 memorize，通常 $p_{\text{prior}} = \mathcal{N}(0, I)$ |

潜变量 $z$ 是从编码器输出的分布 $q_\phi(z|x)$ 中**采样**出来，随机采样“不可导”
所以Reparameterization：$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$
- 噪声 $\epsilon$ 并不代表图像，它代表的是“容错空间”，负责局部连续性；
- $\mu_\phi(x) + \sigma_\phi(x)$负责**特征定位**
- 先验负责**全局结构**


## 4. Discrete Diffusion
# CTMC (continuous-time Markov chains)

--- start-multi-column: CTMC
```column-settings
number of columns: 3
border: off
shadow: off
```

> [!info] General CTMC — Kolmogorov Forward Eq
>
> **KFE** *(master equation for Markov chains)*:
> $$\frac{d}{dt} p_t(x) = \sum_{y} Q_t(x \mid y)\, p_t(y)$$
>
> $Q_t$ = rate matrix（生成元），$Q_t(x \mid y)$ = transition rate from $y \to x$。

--- end-column ---

> [!info] Marginal Distribution
>
> **Marginalization**:
> $$p_t(x) = \sum_{z \in S} p_t(x \mid z)\, p_{\text{data}}(z)$$
>
> **Derivative through marginalization**:
> $$\frac{d}{dt} p_t(x) = \sum_{z \in S} \left( \frac{d}{dt} p_t(x \mid z) \right) p_{\text{data}}(z)$$
>
> $S$ = state space，$p_{\text{data}}$ 与 $t$ 无关，求导可穿入积分号。

--- end-column ---

> [!info] Conditional Distribution
>
> **Conditional KFE** *(each conditional path)*:
> $$\frac{d}{dt} p_t(x \mid z) = \sum_{y \in S} Q_t(x \mid y, z)\, p_t(y \mid z)$$
>
> **Combine into marginal KFE**:
> $$\frac{d}{dt} p_t(x) = \sum_{z \in S} \left[ \sum_{y \in S} Q_t^z(x|y) p_t(y|z) \right] p_{data}(z)$$
> $$\frac{d}{dt} p_t(x) = \sum_{y \in S} \left[ \sum_{z \in S} Q_t^z(x \mid y)\, p_t(y \mid z)\, p_{\text{data}}(z) \right]$$
>where
> $Q_t(x|y) = \sum_{z \in S} Q_t^z(x|y)  \frac{p_t(y|z) p_{data}(z)}{p_t(y)}$
>
>
> 条件 KFE $\to$ 乘 $p_{\text{data}}(z)$ $\to$ 对 $z$ 求和 $\to$ 恢复边际 KFE

--- end-multi-column

为了让边缘转移速率 $Q_t(x|y)$ 正确驱动 KFE 所描述的演化过程，从而完美模拟数据生成
神经网络需要能精准地猜出 $z$（即学好后验概率）

在每一个时刻 $t$，我们把带噪声的序列 $x$ 丢进神经网络。神经网络的任务是：**猜出原始数据 $z$ 在第 $j$ 个位置上到底选了哪个词
$$L_{DFM}(\theta) = E \left[ \sum_{j=1}^d -\log p^\theta_{1|t}(z_j | x) \right]$$


---

### Comparison
| **模型类型**          | **空间** | **微观轨迹 (Micro)**  | **宏观演化 (Macro)** | **数学工具** |
| ----------------- | ------ | ----------------- | ---------------- | -------- |
| **Flow Matching** | 连续     | **确定性** (沿着向量场滑行) | 确定性 (流)          | **ODE**  |
| **Diffusion**     | 连续     | **随机性** (受布朗运动干扰) | 确定性 (分数演化)       | **SDE**  |
| **Discrete FM**   | 离散     | **随机跳跃序列** (瞬间跳变) | **确定性** (概率密度流)  | **CTMC** |
