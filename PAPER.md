# StateSpaceDiffuser: Bringing Long Context to Diffusion World Models

**Nedko Savov**¹†, **Naser Kazemi**¹, **Deheng Zhang**¹, **Danda Pani Paudel**¹  
**Xi Wang**¹·²·³, **Luc Van Gool**¹

¹ INSAIT, Sofia University "St. Kliment Ohridski" &nbsp;&nbsp; ² ETH Zurich &nbsp;&nbsp; ³ TU Munich

†Corresponding author: nedko.savov@insait.ai

_Preprint. Under review._

---

## Abstract

World models have recently gained prominence for action-conditioned visual prediction in complex environments. However, relying on only a few recent
observations causes them to lose long-term context. Consequently, within a few steps, the generated scenes drift from what was previously observed,
undermining temporal coherence. This limitation, common in state-of-the-art world models, which are diffusion-based, stems from the lack of a lasting
environment state.

To address this problem, we introduce StateSpaceDiffuser, where a diffusion model is enabled to perform long-context tasks by integrating features
from a state-space model, representing the entire interaction history. This design restores long-term memory while preserving the high-fidelity
synthesis of diffusion models.

To rigorously measure temporal consistency, we develop an evaluation protocol that probes a model's ability to reinstantiate seen content in extended
rollouts. Comprehensive experiments show that StateSpaceDiffuser significantly outperforms a strong diffusion-only baseline, maintaining a coherent
visual context for an order of magnitude more steps. It delivers consistent views in both a 2D maze navigation and a complex 3D environment. These
results establish that bringing state-space representations into diffusion models is highly effective in demonstrating both visual details and
long-term memory.

Project page: https://insait-institute.github.io/StateSpaceDiffuser/

---

## 1 Introduction

World models have gained popularity for the production of visual consequences of given past observations and actions. These models can learn to
generate environment observations entirely by training on many interactions with the environment. Simply by observation, they are capable of handling
complex environments, such as car driving [27, 29, 41, 66], 3D virtual environments [77, 1, 19], platformer games [8, 68], ego-centric action videos
[82], or navigation [2, 84]. They enable interactivity without the burden of hand-coding complex environments, but also offer feature representations
for robotics and reinforcement learning agents for planning.

For long interaction with world models, it is essential that the generated video remains consistent with previously observed or generated content.
Revisited areas should preserve their appearance, and objects observed again should keep their properties. However, as shown in Fig. 1, current
high-fidelity world models, which are mostly based on diffusion, cannot preserve context outside of a short time window, most often directly limited
by their input window size [82, 1, 19, 8]. This leads to an increasing drift in content over time, where earlier information is forgotten or
overwritten. The inability to retain persistent memory of the environment poses a major challenge, especially for real-world applications such as
agent planning and virtual interaction, where coherent, temporally consistent environments are essential. Therefore, in this work our task is to stay
consistent with a long history of past inputs, even if generating a single future frame. This is in contrast to long generation which focuses on
producing extended realistic sequences, even without prior context.

To improve content consistency in diffusion-based world models, we make use of a persistent long-context representation. Specifically, we leverage
features from a discrete state-space model (Mamba), which has been shown to be very effective at capturing long-term context in prior work [20, 31].
We summarize our system in Fig. 2. Although these models were previously applied to language and visually simple environments [12], our goal is to
preserve the long-term context in modern diffusion-based world models, targeting environments with higher visual complexity such as CSGO [60]. In
contrast, other state-based models, such as those using LSTMs or GRUs [38, 39, 40], have limited generative capacity and are mainly used for agent
planning. Our approach combines the strong generative power of diffusion models with the long-term context tracking of state-space representations.

Importantly, the state-space model (SSM) is computationally efficient, which allows it to process arbitrarily long sequences. This is achieved by
maintaining a compact state that is updated at every sequence step. During training, SSMs have linear complexity in sequence length [31], further
improved by parallelization. Unlike them, CNNs have a fixed receptive depth, and transformers characterize with a heavy quadratic complexity. At
inference, in a streaming fashion, the SSM can be executed with constant per-step latency and constant memory footprint, whereas transformers and
CNNs, at best, still grow linearly per step. As we show in test time, our proposed model scales far beyond its trained horizon, while the SSM
contributes less than 2% of the total inference compute of the full model.

Our proposed model, StateSpaceDiffuser, summarized in Fig. 2, consists of a state-space model that operates over the long sequence, and a diffusion
model – conditioned on both a short window of observations and state-space model features. The latter enables the diffusion branch to generate the
content of the next frame conditioned on a long context rather than the last few frames.

To evaluate the consistency of the generated long context of StateSpaceDiffuser, we design and develop an evaluation framework that involves
navigating environments to then return back to the initial position. We evaluate on two environments: (1) A simpler 2D maze environment (MiniGrid), in
which we establish the presence or absence of memory ability by remembering the maze layout given partial observations. And (2) a 3D first-person
shooter game (CSGO), which serves to show the performance of our method on a visually challenging interactive environment with many factors at play.
Our quantitative and qualitative evaluation results show that StateSpaceDiffuser produces content significantly more consistent with a long history
than a diffusion-only method. Evaluation in the maze environment yields **51.9% PSNR improvement over the baseline on average** (56.3% improvement on
the most memory challenging cases). A user study confirms that our method produces images closer to previously observed content in the CSGO dataset
compared to baselines. More details are shown in Sec. 5.

**Our contributions are as follows:**

- We propose StateSpaceDiffuser, which integrates a state-space model with a diffusion model for visual world modeling. It is capable of generating
  consistent content in long-horizon generation, with almost no extra computational cost.

- We develop an evaluation protocol to test the content preservation abilities of a world model and perform extensive evaluations of world models on
  long-horizon generation tasks.

- Our evaluation shows a significant quantitative improvement and a strong user preference over the baseline in the case of long contexts.
  Furthermore, our studies attribute the improvements to our model design and confirm generalization to longer contexts.

---

## 2 Related Work

### 2.1 World Models

**Generative environment models.** Initially developed as imagination-based models for training model-based reinforcement learning (MBRL) agents [13,
38, 40, 70], world models have evolved into powerful generative systems that condition on actions to produce future frames [11, 41, 57, 83]. Early
work by [36] demonstrates that training a recurrent latent dynamics on VAE image representations can enable agents to plan in imaginative rollouts.
Extensions such as SimPle [47] and Dreamer [37] refine this approach by improving reconstruction quality and stability, culminating in DreamerV2 and
DreamerV3 [39, 40] – systems that achieve human-level performance on Atari and demonstrate the ability to generalize across diverse domains. More
recent efforts, such as IRIS [58], TWM [64], STORM [86], and DayDreamer [79], employ Transformer-based hybrid backbones and focus on sample
efficiency, long-horizon coherence, or robotic control. However, many of these methods rely on discrete latent tokens and relatively short contexts,
which limits visual fidelity in complex scene motion or when extended rollouts are required.

World models are also central to realistic video generation conditioned on actions. Genie [8] leverages a video tokenizer and a Latent Action Model
for dynamic next-frame generation, whereas GAIA-1 [46], GAIA-2 [66] tackle autonomous driving by autoregressively predicting image tokens from
multi-modal inputs. Recent works highlight broader applicability and complex generative capabilities. DINO-WM [87] uses pretrained visual features for
zero-shot planning, GameFactory [84] adapts game environment actions to realistic environments, while allowing video generation control by periodical
text instructions. Both illustrate how world models can transcend traditional RL frameworks and support open-ended content creation.

**Diffusion-based approaches.** Parallel to these developments, diffusion models [73, 44, 75] have emerged as a powerful class of generative methods
for high-fidelity image and video synthesis. They have been applied to text-to-video [71], space-time video generation [3], and broad world simulation
tasks [6]. Within MBRL, DIAMOND [1] uses a diffusion model to generate high-quality frames for Atari, making for playable environments and enhancing
agent performance. Methods like Pandora [80] and LCT [35] generate video based on periodic text instructions. Nonetheless, current diffusion-based
world models, typically transformer-based, condition on only a short window of past frames to handle the quadratic computational complexity, making
long-horizon dependencies difficult to maintain. This makes it challenging to maintain long-horizon dependencies.

### 2.2 Sequence Modeling

**RNNs and Transformers.** Sequential modeling has historically been dominated by recurrent neural networks (RNN) such as LSTM and GRU [45, 15, 17],
which process input tokens step by step and are able to handle moderate contexts. However, RNNs often struggle with extremely long sequences due to
vanishing gradients and limited memory capacity [59]. Transformers [78] addressed these issues by employing self-attention, making them effective in
capturing long-range dependencies. Beyond world modeling, Transformers have become the backbone for a broad range of tasks, including language
modeling [21, 7, 63] and computer vision [22, 43, 9, 28], due to their ability to handle global context. Various Transformer variants have attempted
to reduce the quadratic cost of self-attention for long sequences [50, 16, 4, 85, 14]. Vision-specific models like Swin [56] or MViT [24] adopt
hierarchical or local attention, yet scaling them to long video horizons remains computationally prohibitive.

Previously, DFoT [10] addressed the ability for long future prediction. However, the long-context consistency problem has only been recently addressed
by a few concurrent works. [51, 74, 81] improve context abilities by proposing strategies to sample a number of historical observations to use as
conditioning. Instead, our approach involves summarizing information from the entire history automatically through state-space models.

**State-Space Models (SSMs).** As an alternative, SSMs [5, 53, 62, 76, 61] can process sequences in linear time by learning continuous dynamics in a
latent state. Representative structured state-space models include S4 [33, 34] and H3 [18] that generalizes the recurrence in Linear Attention [49].
S4, S5 [72], and S6 [52] leverage carefully designed operators (e.g., HiPPO matrices [32]) to efficiently capture long-range dependencies. Mamba [31]
introduces selective gating to improve expressiveness without sacrificing linear scalability. S4WM [20] has shown that applying SSMs as world models
shows promise for maintaining coherence over hundreds of imagined steps while preserving computational tractability.

**Hybrid Architectures.** As Transformers excel at local interaction with low computational cost and SSMs can capture long-horizon dependencies
efficiently, hybrid designs have been proposed for vision tasks. MambaVision [42] incorporates state-space models into a transformer, and Dimba [26],
DiS [25] – into a diffusion network backbone, for computationally cheaper image discriminative and generative tasks, operating on image patches. [54]
modify the softmax in attention to emulate a forget gate and improve transformer context abilities. MambaVLT [55] and Samba [69] exploit state-space
models for better object tracking with long-range consistency.

---

## 3 Data

We design an experimental protocol to evaluate long-term content consistency in diffusion world models, comprising of three experimental setups with a
rising level of complexity, based on a controlled maze environment (MiniGrid) and a complex 3D first-person environment (CSGO).

We create a dataset based on the partially observed MiniGrid maze environment [12]. In this setup, each maze consists of a grid where each cell can be
a wall, an empty space, or a colored marker. Markers act like empty spaces, but are visually distinct. An agent navigates the maze, but at each time
step, it only sees a 7×7 window centered around itself rather than the full 85×85 maze (see Fig. 6 (b) for an example). We use a modified version of
MiniGrid with randomly generated mazes, allowing us to adjust the size, wall complexity, and number of color-coded markers. In each episode, the agent
is tasked with visiting a sequence of 40 random markers via the shortest path. Once halfway through the episode, the agent stops following the path
and retraces its steps back to the starting point. Each episode is 100 steps long (50 forward, 50 backwards). We evaluate on different context lengths
by selecting subsequences around the long sequence center. Notably, the second half of each sequence depends heavily on the model's ability to recall
earlier frames, making it ideal for testing long-context reasoning.

We also design a simplified dataset called MiniGrid Simple, consisting of just 34 samples without walls and a single marker placed behind the starting
position. The agent moves three steps forward and three steps back, returning to its initial position. Since the context window of our baseline is
just 4 steps, this setup provides a minimal but effective test of long-term recall. We use this to compare the performance of our baseline and
state-space-enhanced models in reconstructing the marker color.

To evaluate our method in a more visually complex setting, we use CSGO [60], a dataset of human gameplay in a 3D first-person shooter. It includes 51
action types, such as 23 rotational commands, 4 movement directions, jumping, and various special actions (e.g., firing, changing weapons). To adapt
the dataset for long-context testing, we create mirrored sequences: for each original sequence, we append its reversed version, ensuring that actions
are also reversed. We use a corresponding one-hot encoding (e.g. turning left becomes turning right), or create a new one if a correspondence does not
exist (e.g. jump, shoot). This setup forces the model to rely on information from earlier in the sequence when generating later frames.

---

## 4 Methodology

Given a sequence of environment interactions $a_1, a_2, ..., a_{T-1}$, the resulting observations $I_1, I_2, ..., I_T$ (with initial frame $I_1$) and
the current action $a_T$, the objective of a world model $\mathcal{F}(\cdot)$ is to produce the next image
$I_{T+1} = \mathcal{F}([I_1, ..., I_T], [a_1, ..., a_T])$. Recently, the best-performing generative architectures for modeling $\mathcal{F}(\cdot)$
are diffusion models based on transformers or UNet. In training, transformers are computationally intensive – $O(T^2)$. CNNs have fixed receptive
fields and are ill-fitted to long-term dependencies. Therefore, these models take a short history window of observations:
$I_{T+1} = \mathcal{F}([I_{T-K+1}, ..., I_T], [a_{T-K+1}, ..., a_T])$. (e.g. $K = 4$ [1, 77], $K = 16$ [8]). With a long and growing sequence, the
short history causes the loss of long-term temporal coherence. Instead, we propose to efficiently process the long sequence ($O(T)$) with a model
designed for this purpose – a state-space model. Such models maintain and update a state with each sequence step, and the state serves as a summary of
the sequence so far. Extracting long-context features in this way and integrating them into the diffusion pipeline yields the proposed model –
StateSpaceDiffuser.

### 4.1 StateSpaceDiffuser Architecture

Our architecture is shown in Fig. 3. It is conceptually divided into two branches: Long-Context Branch and Generative Branch. The Long-Context Branch
preserves information over long sequences, and the Generative Branch uses this context to render high-quality images.

**Long-Context Branch.** In contrast to transformer and CNN architectures, state-space models (SSMs) are designed specifically to efficiently process
long sequences. Although SSMs are generally designed for continuous input signals, discrete SSMs maintain an internal state representation $h$ that is
updated with each time step $t$ in an input sequence of one-dimensional feature vectors $f_1, ..., f_T$, through the parameter matrices $A$, $B$ and
$C$, which are learned in training time:

$$h_t = Ah_{t-1} + Bf_t, \quad m_t = Ch_t$$

We denote a state-space model with $m_1, ..., m_T = \mathcal{M}(f_1, .., f_T)$, with $m_t$ denoting the model's output. To bring it into the world
model setting, we define $f_t$ to be a compact feature representation of $I_t$ and train a model that predicts future observations:
$\hat{f}_2, ..., f_{T+1} = \mathcal{M}([f_1, a_1], .., [f_T, a_T])$. It is common in existing work to apply SSMs at the patch or image token level as
common in previous work [42, 26]. Instead, we avoid conflating spatial and temporal dependencies by temporally processing full frames. Each frame is
encoded into a single compact feature $f_t$ used as a single step of our sequence. The encoding is obtained by the continuous Cosmos tokenizer [23]
with scale 16 (CI16). The resulting patch tokens (dimension 16) are flattened to form the single feature vector $f_t$ per image. Alongside it, we
incorporate a discrete action $a_t$, which indexes a learnable embedding of dimension 16. We concatenate $f_t$ and $a_t$ to create the input at each
step. We adopt the Mamba architecture due to its dynamic selection mechanism and efficient parallelism, which led to superior performance compared to
other SSM variants.

One key benefit of state-space models is their computational efficiency. As only a single state is maintained in inference, memory remains constant
regardless of context length. As the same update is applied linearly on a sequence, computational complexity remains $O(T)$. When presented with a
growing sequence, Mamba only updates the state from the previous step, making for a constant per-step latency. In contrast, CNNs and transformers do
not maintain a state and have to reprocess the growing sequence at every step. In training, Mamba further parallelizes the sequence processing.

Provided a long input sequence of high-resolution images and corresponding actions, the model predicts the Cosmos features corresponding to the next
observation with an MSE loss. In those features, we expect relevant to the time-step information, recalled from the sequence. While context cues are
to be preserved, the state is low-dimensional, and this model is not generative. Therefore, the generative branch, designed for the generation of
high-quality images, is intended to render the final output.

**Generative Branch.** To generate high-quality images in complex environments, we employ a diffusion model. Our choice is the DIAMOND world model
[1], a UNet-based EDM diffusion model [48] designed for visual prediction in sequential environments. Therefore, our Generative Branch conditions on
only four low-resolution frames and their corresponding actions, represented as 512-dimensional action embeddings. Despite this minimal context, it
can produce high-quality predictions with just three denoising iterations per output frame. The architecture consists of two diffusion models: a
primary model that predicts the next observation at a low resolution, and a secondary upsampler that refines these predictions to a resolution of
$280 \times 150$. As the model predicts one frame at a time, generating longer sequences is achieved through a sliding-window strategy, and each newly
generated frame is appended to the input history for the next prediction. In isolation, this strategy causes a short-context limitation to this model.

**Fusion of features.** To address the context limitation of the Generative Branch, we build a fusion module to integrate state-space features into
it, in order to provide long-context information. To that end, we process the entire sequence with the Long-Context branch and obtain the last 4
output features $\hat{f}_t$. We fuse those features with the corresponding action embeddings from the Generative Branch. These features are first
normalized and then passed through a two-layer MLP with SiLU activation, where the input size matches the feature dimensions. Similarly, the action
embeddings, perturbed with noise, are processed by an MLP with the same architecture. To form the final conditioning vector, we concatenate the
outputs of the two MLPs. Empirically, we discovered that processing the memory and action conditions independently before concatenation yielded better
performance than fusing them earlier in the pipeline.

### 4.2 Training Protocol

At each step, a batch consists of sequences of actions, reversed actions and observations. Training is performed in two stages. Firstly, the
Long-Context Branch is trained on long context – length 50 or 16. The produced features decode to images with artifacts, but with important context
cues. Then, freezing the Long-Context Branch, we train the Generative Branch, conditioned on the compressed long-context features, with a sequence
size of 4. This branch produces the final high-quality images with the correct context. Training details are given in App. A.

We found that this two-stage training is crucial for stability. Direct end-to-end training is unstable, as diffusion gives noisy gradients to the SSM,
and the SSM gives constantly changing features to diffusion. In turn, diffusion learns to ignore the SSM features. Therefore, stable features of a
pretrained SSM worked best in this architecture. Moreover, the training separation enabled to swap out in test time the Long-Context Branch with
another independently trained model, without having to further fine-tune the heavier Generative Branch.

---

## 5 Experiments

### 5.1 Experimental Setup

**Baselines.** We establish two baselines. The first is a pure diffusion model without state-space features: the DIAMOND model. Our second baseline is
the State-Space World Model. It is the Long-Context branch of StateSpaceDiffuser and its training is equivalent to the first stage of training, as
described in Sect. 4.2. At inference time, the predicted feature $f_t$ is decoded into an image $I_t$ using the decoder from the Cosmos tokenizer. In
App. B.2 we present comparisons of sequence models to solidify our choice of Mamba as our backbone.

This model enables us to assess the memory capacity of state-space models (SSMs) in sequential visual prediction. Although its outputs tend to be
blurry and contain artifacts in complex scenes, due to the absence of a variational component and limited generative expressiveness compared to modern
diffusion models, the SSM exhibits a strong ability to model long sequences and retain information from earlier in the trajectory. The strengths and
shortcomings observed in this baseline directly inform and motivate the design of StateSpaceDiffuser.

**Testing Protocol.** Our evaluation protocol matches our mirrored action setup – we take $n$ actions and $n$ reverse actions, and expect to generate
the same observations for the second half of the sequence as seen in the first. On MiniGrid, we have a fixed sizeable visual difference per step,
while for CSGO continuous motion often results in small per-step changes. Therefore, in MiniGrid, we generate one frame in the future at a time, while
in CSGO we sequentially generate the whole second half of the sequence. On MiniGrid we evaluate with PSNR and SSIM on varying future horizons – the
further in the sequence, the longer the memory required. In CSGO we perform a user study, more aligned to the visual complexity of the environment. We
motivate this difference with the known mismatch between perceived quality and fidelity metrics in continuous video [65, 67, 30] (App. D.3). Although
the baseline performs well when context is not essential, our protocol exposes its inability to model long-term context, resulting in degraded quality
in this scenario.

### 5.2 Results and Analysis

**Simple MiniGrid Evaluation.** In this experiment, we test the recall ability of the baseline, the State-Space World Model and StateSpaceDiffuser, on
a simple toy setup, as described in Sect. 3. We train and test on the same set of 34 samples. The goal is to recall a color at the final frame from
the first frame in the sequence with a length of 7 frames. Two random samples (colors) from the results are shown in Fig. 4, with the corresponding
model predictions. With input size 4, the baseline processes the sequence in a sliding window fashion and, as within the 3 steps the color information
is lost, it cannot reconstruct the correct color. Despite the small training set size, the baseline fails because of a lack of long-context abilities.
In contrast, our State-Space World Model, based on a computationally efficient state-space model, is able to predict the correct color. Finally, it is
demonstrated that our StateSpaceDiffuser is also able to recall the correct content by effectively combining both paradigms. Notably, our methods
perform equivalently on a context length of 50 frames – when predicting the 51st, StateSpaceDiffuser recalls the color from 50 steps ago.

**Forward-Backward Evaluation on MiniGrid.** We compare the long context abilities of our diffusion (DIAMOND) and state-space (State-Space World
Model) baselines in our MiniGrid test set. We evaluate our models trained on context length 50 on context lengths 16 and 50 (demonstrating
generalizability). We follow the protocol outlined in Sect. 5.1. To evaluate, we compute the Peak Signal-to-Noise Ratio (PSNR) for each predicted
frame in the reverse trajectory, reporting both the mean score and the PSNR at the final time step, which requires the longest-term memory. As shown
in Tab. 1, our model significantly outperforms both baselines, particularly at the end of the sequence, where successful recall of the first frame is
critical. This highlights the model's ability to retain and reinstantiate long-term visual context. In App. B.4, B.5, we show the stability and
robustness of these results, in App. B.1 – performance gain analysis over computational cost. Compared to the State-Space World Model, our method
achieves higher fidelity output, benefiting from the superior generative capacity of the Generative Branch (examples – in App. C.1, C.2). Fig. 6 (b)
presents example rollouts generated by our model and the diffusion-only baseline. In MiniGrid, predictions are made one step at a time using the
ground truth sequence. As a result, most content is carried over from the previous frame, with only the newly revealed area requiring inference. Our
method excels at filling in these newly revealed regions, even when the relevant context originates far back in the sequence. In contrast, the
diffusion baseline struggles to recover such long-range dependencies.

**Recall Across a Context Length.** In this experiment we study the accuracy of our models over the varying context length of the forward-backward
evaluation on MiniGrid. When predicting future observations, the last frame's content depends on the first frame's content, and the further back we go
in the sequence the smaller the context length required for a good reconstruction. This is a direct consequence of the mirror style of the
observations in our setup. In Fig. 7 we show the PSNR at each predicted time step. The first few predicted frames are easily predicted by all models
as the solution falls within the short input window. However, performance for the diffusion baseline quickly falls as no form of information is
preserved from the long context, while a state-space model is able to harvest this information. Our StateSpaceDiffuser model gets the best of both
worlds – long-context awareness and high-fidelity predicted images, and performs the best.

**Forward-Backward Imagination Evaluation on CSGO.** Similarly to MiniGrid, we evaluated the recall abilities of our model on the CSGO dataset, a
visually complex environment in a 3D world. In CSGO most actions are gradually executed over a sequence, and there is a compounding effect on content
change (e.g. jump unfolds over many frames). For a high impact evaluation we decide to give only the first half of the sequence and continuously
produce the second half (reverse) by feeding generated frames. As actions are motions at varying levels, the final frame may contain the correct
content memorized but with low PSNR, as the camera position and scene geometry might be slightly shifted. Therefore, instead of fidelity metrics we
perform a user study where the 12 participants judge whether images produced by StateSpaceDiffuser are closer in content to the ground truth compared
to the diffusion baseline (details – in App. D.4). Our rating is in the range $[-1, 1]$, with 0 being borderline, -1 – preference toward the baseline,
1 – preference for StateSpaceDiffuser. The results shown in Fig. 5 demonstrate a clear preference of the users for StateSpaceDiffuser over the
baseline for both prediction in the 15th frame (rating **0.20**) and 17th (last) frame (rating **0.24**). Fig. 6 (a) shows a sample of CSGO
imagination at different time steps, demonstrating that while the baseline fails to recall the correct content, the StateSpaceDiffuser correctly
produces the details. (More in App. D.1)

**State Features Ablation.** We study the utility of the state-space features provided to the Generative Branch in our StateSpaceDiffuser model. We
take a trained model and perform a MiniGrid evaluation by replacing the output features of the Long-Context Branch with zeros before passing them to
the Generative Branch. In Tab. 1 we show that this causes the performance to quickly drop even below baseline performance, clearly demonstrating that
the features are highly utilized. In Fig. 8 we demonstrate the same effect on CSGO. Without state features, the model hallucinates; without diffusion,
the state-space model remembers but produces poor visual quality. (More in App. B.7), B.8)

**Generalization to Longer Context.** In this experiment we show that StateSpaceDiffuser operates on much longer contexts without finetuning. We
evaluate our model trained on context length 50 on lengths 100 and 150 using a new MiniGrid test set with longer sequences. Tab. 2 shows that
StateSpaceDiffuser successfully generalizes to longer context, keeping a significant gain over the baselines. Analogously, in App. B.3, we show
generalization from context length 16 to length 50.

### 5.3 Strengths, Limitations and Scalability

Apart from the already established generalization across context length, via extra experiments, we find that StateSpaceDiffuser is able to generalize
across visual complexity (App. B.6) and can recover from strong motion artifacts (App. D.2). Our model can recover from input noise in future steps,
but is clearly affected by it on the current steps (App. B.5). Our lightweight StateSpaceDiffuser was trained under a fixed compute budget. The
lightweight diffusion decoder (no large pretrained backbone) can yield visual artifacts in long rollouts. Replacing the decoder with a better, larger
one, can improve visual sharpness without changing the method. Our lightweight single-layer Long-Context Branch compresses the context into a
low-dimensional state (256), which can cause loss of detail in extended rollouts, especially in complex environments (App. D.2). Scaling the SSM
(state dimension/heads/parameters/layers) is expected to reduce high-frequency decay over time. The separation in training enables separately scaling
each branch before combining them.

---

## 6 Conclusion

We introduced **StateSpaceDiffuser**, a hybrid model that combines state-space representations with diffusion to enable long-horizon visual world
modeling. By decoupling global context modeling (via a state-space backbone) from high-fidelity synthesis (via diffusion), our model retains global
context over many steps at essentially no additional computational cost. The resulting representation alleviates the drift and inconsistency that
plague conventional diffusion-only systems in long sequences.

Experiments on MiniGrid and CSGO validate our method's consistency and fidelity across long sequences. In the forward-backward protocol with horizon
50, StateSpaceDiffuser improves average PSNR by **51.9%** over the diffusion baseline and achieves a final-frame PSNR of **39.32** versus **25.14**
for DIAMOND on a long context length of 50 frames. Human raters also favor our generations for long-context consistency (Fig. 5).

Our results establish state-space diffusion as a scalable and consistent solution for long-context visual generation. We believe that bridging
state-space reasoning with diffusion generation is a promising direction for robust, long-horizon world modeling, and we hope this work lays a solid
foundation for future research in temporally coherent visual prediction.

---

### Tables

**Table 1: MiniGrid Quantitative Evaluation of Long-Context Awareness.** Our StateSpaceDiffuser outperforms the baselines.

| Model                                | Avg. PSNR↑ | Fin. PSNR↑ | SSIM↑    |
| ------------------------------------ | ---------- | ---------- | -------- |
| _Context Length 16_                  |            |            |          |
| DIAMOND                              | 27.13      | 25.44      | 0.95     |
| State-Space World Model              | 33.40      | 33.17      | 0.96     |
| StateSpaceDiffuser (Ours, w/o state) | 23.68      | 20.95      | 0.92     |
| **StateSpaceDiffuser (Ours)**        | **41.01**  | **40.55**  | **0.98** |
| _Context Length 50_                  |            |            |          |
| DIAMOND                              | 26.13      | 25.15      | 0.95     |
| State-Space World Model              | 32.64      | 32.44      | 0.96     |
| **StateSpaceDiffuser (Ours)**        | **39.68**  | **39.32**  | **0.98** |

**Table 2: Generalization to Longer Context.** Our model, trained on context length 50, generalizes to longer sequences (context 100 and 150).

| Model                         | Avg. PSNR | Fin. PSNR | SSIM     |
| ----------------------------- | --------- | --------- | -------- |
| _Context Length 100_          |           |           |          |
| DIAMOND                       | 26.39     | 26.24     | 0.95     |
| State-Space World Model       | 31.65     | 30.89     | 0.96     |
| **StateSpaceDiffuser (Ours)** | **37.99** | **35.87** | **0.98** |
| _Context Length 150_          |           |           |          |
| DIAMOND                       | 24.35     | 24.20     | 0.94     |
| State-Space World Model       | 27.93     | 26.98     | 0.94     |
| **StateSpaceDiffuser (Ours)** | **30.75** | **28.93** | **0.96** |
