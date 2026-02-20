# Days 13–15: ML Foundations

> Classical ML, deep learning, and math that interviewers assume you know — Feb 2026

---

## Why This Guide Exists

```
Interview reality for SDE2 + AI/GenAI roles:

Round 1: DSA / Coding         ← You have LeetCode ✅
Round 2: System Design         ← You have Grokking SD ✅
Round 3: ML / AI Knowledge     ← Days 1-12 cover GenAI ✅
Round 4: ML Fundamentals       ← THIS GUIDE fills the gap ⚠️

The ML Fundamentals round is where they ask:
  "What's bias-variance tradeoff?"
  "How does gradient descent work?"
  "When would you use XGBoost vs a neural network?"
  "Explain precision vs recall"
  "How do transformers work under the hood?"

Even for GenAI-focused roles, 70%+ of interviewers will ask
at least 2-3 classical ML questions to test foundations.
```

---

## Table of Contents
1. [Day 13 — Classical ML for SDE2 Interviews](#day-13--classical-ml-for-sde2-interviews)
2. [Day 14 — Deep Learning & Transformers](#day-14--deep-learning--transformers)
3. [Day 15 — ML Metrics, MLOps & Production ML](#day-15--ml-metrics-mlops--production-ml)
4. [Interview Cheat Sheet — ML Buzzwords](#-interview-cheat-sheet--ml-buzzwords)
5. [ML System Design Patterns](#-ml-system-design-patterns)
6. [Glossary](#glossary)

---

# Day 13 — Classical ML for SDE2 Interviews

## 1. The ML Landscape — What's What?

```
Machine Learning
  ├── Supervised Learning (labeled data → predict)
  │     ├── Classification (predict category)
  │     │     ├── Logistic Regression
  │     │     ├── Decision Trees / Random Forest
  │     │     ├── SVM
  │     │     ├── Naive Bayes
  │     │     └── XGBoost / LightGBM
  │     └── Regression (predict number)
  │           ├── Linear Regression
  │           ├── Ridge / Lasso
  │           ├── Decision Trees
  │           └── XGBoost / LightGBM
  │
  ├── Unsupervised Learning (no labels → discover patterns)
  │     ├── Clustering (group similar items)
  │     │     ├── K-Means
  │     │     ├── DBSCAN
  │     │     └── Hierarchical
  │     ├── Dimensionality Reduction
  │     │     ├── PCA
  │     │     └── t-SNE / UMAP
  │     └── Anomaly Detection
  │           └── Isolation Forest
  │
  ├── Semi-Supervised (few labels + lots of unlabeled)
  ├── Self-Supervised (create labels from data itself — how LLMs learn!)
  └── Reinforcement Learning (agent + environment + rewards)
        └── RLHF (used to train ChatGPT!)
```

### Interview Buzzword: "Supervised vs Unsupervised vs Self-Supervised"
Supervised = labeled data (spam detection). Unsupervised = discover patterns without labels (clustering). Self-supervised = create your own labels (LLMs predict next token). GenAI is built on self-supervised pre-training.

### Resume Keyword
"ML fundamentals — supervised, unsupervised, and self-supervised learning"

---

## 2. The Algorithms You MUST Know (Ranked by Interview Frequency)

### Tier 1 — Asked in 80%+ of interviews

#### Linear Regression (The Foundation)
```
What: Find the best line through your data points
Math: y = wx + b (weight × feature + bias)
Goal: Minimize squared errors (MSE)

Training: Gradient descent or closed-form solution (Normal Equation)

        y
        │    •       /
        │  •       /   ← best fit line: y = 0.5x + 1
        │       /  •
        │     / •
        │   /
        └──────────── x

When to use:
  ✅ Continuous output (price, temperature, score)
  ✅ Features have linear relationship with output
  ✅ You need interpretability (coefficients tell you feature importance)
  ❌ Non-linear relationships (use tree-based models)
```
Interview Q: "What assumptions does linear regression make?"
Four: (1) Linear relationship, (2) Independent errors, (3) Homoscedasticity (constant variance), (4) Normally distributed errors. Check residual plots. If violated, try polynomial features or tree-based models.
Four key assumptions: (1) Linear relationship between features and target, (2) Independence of errors, (3) Homoscedasticity — constant variance of errors, (4) Normally distributed errors. In practice, I check residual plots. If assumptions are violated, I'd consider polynomial features or switch to tree-based models.

#### Logistic Regression (Classification, Despite the Name!)
```
What: Linear regression + sigmoid function → probability [0,1]
Math: P(y=1) = sigmoid(wx + b) = 1 / (1 + e^(-z))

     P(y=1)
     1.0│            ─────────
        │          /
     0.5│─ ─ ─ ─/─ ─ ─ ─ ─ ← decision boundary
        │      /
     0.0│─────
        └──────────────────── z (linear combination)

Output: Probability → threshold (usually 0.5) → class label
Loss function: Binary Cross-Entropy (log loss)
Optimization: Gradient Descent

When to use:
  ✅ Binary classification with interpretable results
  ✅ You need probability outputs, not just class labels
  ✅ Baseline model — always try this FIRST
  ❌ Complex non-linear boundaries
```

### Interview Q: "Logistic regression vs SVM?"
Logistic regression = interpretable, gives probabilities. SVM = maximizes margin, handles non-linear boundaries with kernels. Start with logistic regression as baseline, then try XGBoost.

#### Decision Trees & Random Forests
```
Decision Tree:
  ┌──────────────────┐
  │ Age > 30?         │
  ├────────┬──────────┤
  │ Yes    │ No       │
  │        │          │
  │ Income │ Student? │
  │ >50K?  │          │
  │ Y │ N  │ Y  │ N   │
  │ ✅│ ❌ │ ✅ │ ❌  │
  └────────┴──────────┘

Splitting criteria:
  - Gini Impurity (default in sklearn): measures class imbalance
  - Information Gain (Entropy): measures surprise reduction

Problem: Overfits badly! (memorizes training data)

Random Forest (the fix):
  - Train 100+ decision trees on random subsets of data (bagging)
  - Each tree uses random subsets of features
  - Final answer = majority vote (classification) or average (regression)
  - Reduces overfitting while keeping tree benefits
```

### Interview Buzzword: "Bagging / Ensemble Learning"
Random Forest uses bagging — it trains many decision trees on bootstrapped samples with random feature subsets. This reduces variance without increasing bias. The 'wisdom of crowds' effect: individual trees overfit, but together they generalize. It's one of the most reliable out-of-the-box algorithms.

#### XGBoost / Gradient Boosting (The ML Competition King)
```
Boosting (different from Bagging!):
  1. Train a weak model
  2. Look at what it got WRONG
  3. Train the next model to fix those errors
  4. Repeat 100+ times
  5. Final answer = weighted sum of all models

     Model 1: accuracy 60%
       ↓ (focus on errors)
     Model 2: accuracy 70%
       ↓ (focus on remaining errors)
     Model 3: accuracy 78%
       ↓ ...
     Ensemble: accuracy 95%

XGBoost advantages:
  - Regularization (L1 + L2) prevents overfitting
  - Handles missing values natively
  - Parallel tree construction (fast!)
  - Feature importance built-in
  - Still wins most tabular data competitions in 2026

LightGBM: Microsoft's variant, faster on large datasets
CatBoost: Handles categorical features natively
```

### Interview Q: "When would you use XGBoost vs a neural network?"
XGBoost for tabular/structured data (databases, spreadsheets) — it's faster to train, easier to interpret, and usually more accurate on structured data. Neural networks for unstructured data (text, images, audio) where feature engineering is impractical. In 2026, the rule is: tabular → XGBoost, unstructured → neural nets/LLMs.

### Resume Keyword: "XGBoost, Random Forest, gradient boosting for classification & regression"

---

### Tier 2 — Asked in 40%+ of interviews

#### K-Nearest Neighbors (KNN)
```
Idea: To classify a new point, look at its k closest neighbors
      and vote. Majority wins.

No training phase! (lazy learner — stores all data)
Distance: usually Euclidean or cosine similarity

Pros: Simple, no assumptions, works well for small datasets
Cons: Slow at prediction (searches all data), curse of dimensionality

Fun fact: Your Day 4 embeddings + vector search IS basically KNN!
          pgvector's similarity search = KNN with cosine distance
```

#### K-Means Clustering
```
Goal: Group n objects into k clusters

Algorithm:
  1. Pick k random center points
  2. Assign each point to nearest center
  3. Recalculate centers as cluster means
  4. Repeat until stable

     Cluster 1: • • •      Cluster 2: ▪ ▪ ▪
                  •   •                 ▪  ▪
                   ★            ← centers     ★
                  •                     ▪ ▪

Choose k: Elbow method — plot error vs k, pick the "elbow"

Limitation: Must specify k upfront, assumes spherical clusters
Alternative: DBSCAN (density-based, finds arbitrary shapes)
```

#### SVM (Support Vector Machine)
```
Idea: Find the hyperplane that maximizes the margin between classes

     ●  ●     |     ○  ○
       ●    ← margin →  ○
     ●  ●     |     ○  ○
              ↑
        decision boundary

Kernel trick: projects data to higher dimensions where it becomes
              linearly separable (polynomial, RBF, sigmoid kernels)

Good for: Small-medium datasets, high-dimensional data (text)
Mostly replaced by: XGBoost for tabular, neural nets for text
Still relevant: As an interview concept (kernel trick, margins)
```

#### Naive Bayes
```
Based on Bayes' Theorem:
  P(spam | words) = P(words | spam) × P(spam) / P(words)

"Naive" because it assumes features are independent (they're not,
 but it works surprisingly well!)

Fast, simple, great baseline for text classification
Good for: Spam detection, sentiment analysis, document classification
```

### Interview Q: "Walk me through how you'd approach a new ML problem"
Five steps: (1) Understand the problem — is it classification, regression, clustering? (2) EDA — look at data distribution, missing values, class imbalance. (3) Baseline — logistic regression or XGBoost, measure with appropriate metrics. (4) Iterate — feature engineering, hyperparameter tuning, try other models. (5) Evaluate — cross-validation, check for overfitting, measure on held-out test set.

---

## 3. The Bias-Variance Tradeoff (Asked in EVERY Interview)

```
Total Error = Bias² + Variance + Irreducible Noise

HIGH BIAS (Underfitting):
  Model is too simple → misses patterns
  Example: Linear regression on non-linear data
  Training error: HIGH ❌
  Test error:     HIGH ❌
  Fix: More complex model, more features

HIGH VARIANCE (Overfitting):
  Model is too complex → memorizes noise
  Example: Deep decision tree on small dataset
  Training error: LOW ✅
  Test error:     HIGH ❌
  Fix: More data, regularization, simpler model, dropout

THE SWEET SPOT:
  Error
    │\
    │ \  Total Error
    │  \___________/
    │   ╲        ╱
    │    ╲      ╱
    │ Bias²╲  ╱ Variance
    │       ╲╱
    └──────────────── Model Complexity
              ↑
         Sweet Spot
```

### Interview Buzzword: "Bias-Variance Tradeoff"
Bias is error from wrong assumptions (underfitting). Variance is error from sensitivity to training data (overfitting). The goal is the sweet spot. I manage this with regularization (L1/L2), cross-validation, and ensemble methods. Random forests reduce variance via bagging; boosting reduces bias via sequential learning.

---

## 4. Regularization — Fighting Overfitting

```
Overfitting: model memorizes training data (low train error, high test error)

Prevention toolkit:
  ├── L1 (Lasso): adds |w| penalty → drives unimportant weights to exactly 0
  │                → feature selection! tells you which features matter
  │
  ├── L2 (Ridge): adds w² penalty → shrinks all weights toward 0
  │               → keeps all features but reduces their impact
  │
  ├── Elastic Net: L1 + L2 combined → best of both worlds
  │
  ├── Dropout (neural nets): randomly disable neurons during training
  │                          → forces redundancy, prevents co-adaptation
  │
  ├── Early Stopping: stop training when validation error starts increasing
  │
  └── Data Augmentation: create more training examples
                         → images: rotate, flip, crop
                         → text: paraphrase, back-translate, synonym replace
```

### Interview Q: "L1 vs L2 regularization?"
L1 (Lasso) drives weights to exactly zero — it does automatic feature selection. L2 (Ridge) shrinks weights but keeps them non-zero — it handles multicollinearity better. I use L1 when I suspect many irrelevant features, L2 when all features are potentially useful. Elastic Net combines both.

---

## 5. Cross-Validation (How to Properly Evaluate)

```
WRONG: Train on all data, test on same data → inflated metrics!

RIGHT: Hold out data the model never sees during training

K-Fold Cross-Validation (k=5 example):
  Fold 1: [TEST] [Train] [Train] [Train] [Train] → score₁
  Fold 2: [Train] [TEST] [Train] [Train] [Train] → score₂
  Fold 3: [Train] [Train] [TEST] [Train] [Train] → score₃
  Fold 4: [Train] [Train] [Train] [TEST] [Train] → score₄
  Fold 5: [Train] [Train] [Train] [Train] [TEST] → score₅
  
  Final score = average(score₁...score₅)

Variants:
  - Stratified K-Fold: preserves class distribution in each fold
  - Leave-One-Out (LOO): k = n (expensive, low bias)
  - Time-Series Split: always train on past, test on future

Data splitting in practice:
  Train (70%) → model learns patterns
  Validation (15%) → tune hyperparameters
  Test (15%) → final evaluation (touch ONCE!)

⚠️ NEVER use test set for hyperparameter tuning!
   That's "data leakage" → inflated metrics → model fails in production
```

### Interview Buzzword: "Data Leakage"
Data leakage is when information from outside the training set bleaches into the model — either through test set contamination or feature leakage. Common examples: including the target variable indirectly in features, or preprocessing (scaling, encoding) before splitting. I always split first, then preprocess.

---

## 6. Feature Engineering — What SDE2s Are Expected to Know

```
Feature Engineering = transforming raw data into useful model inputs

Common techniques:
  ├── Numerical
  │     ├── Scaling: StandardScaler (mean=0, std=1), MinMaxScaler (0-1)
  │     ├── Log transform: fix right-skewed distributions
  │     └── Binning: continuous → categorical (age → age_group)
  │
  ├── Categorical
  │     ├── One-hot encoding: color → [is_red, is_blue, is_green]
  │     ├── Label encoding: ordinal categories → [0, 1, 2]
  │     ├── Target encoding: replace with mean of target (careful of leakage!)
  │     └── Embedding: learned dense vectors (used in deep learning)
  │
  ├── Text (pre-LLM)
  │     ├── TF-IDF: term frequency × inverse document frequency
  │     ├── Bag of Words: word count vectors
  │     └── Now: LLM embeddings (your Day 4!) replace all of these
  │
  ├── Missing Values
  │     ├── Imputation: mean, median, mode, KNN, model-based
  │     ├── Indicator: add "is_missing" boolean feature
  │     └── XGBoost: handles missing values natively ✅
  │
  └── Feature Selection
        ├── Filter: correlation, mutual information, chi-squared
        ├── Wrapper: recursive feature elimination (RFE)
        └── Embedded: L1 regularization, tree-based importance
```

### Interview Tip
For text features in 2026, I use LLM embeddings instead of TF-IDF — an OpenAI `text-embedding-3-small` call replaces an entire text preprocessing pipeline. But for structured/tabular data, traditional feature engineering still matters — XGBoost on well-engineered features often beats neural networks.

### Resume Keyword: "Feature engineering, data preprocessing, model selection"

---

## 7. Dimensionality Reduction

```
PCA (Principal Component Analysis):
  - Finds directions of maximum variance in data
  - Projects data onto these directions (principal components)
  - Reduces features while preserving most information
  
  100 features → PCA → 10 components (95% variance preserved)
  
  Use for: Visualization, denoising, compression, preprocessing
  Math: eigendecomposition of covariance matrix

t-SNE / UMAP (for visualization):
  - Non-linear dimensionality reduction
  - Great for visualizing high-dimensional data in 2D/3D
  - t-SNE: slower, good for small datasets
  - UMAP: faster, better for large datasets, preserves global structure
  
  Fun connection: Arize Phoenix uses UMAP to visualize embeddings
                  (your Day 6 observability!) — shows embedding drift
```

### Interview Buzzword: "PCA / Dimensionality Reduction"
PCA finds orthogonal axes of maximum variance. I'd use it to reduce a 1000-feature dataset to 50 components that capture 95% of variance — faster training, less overfitting. For embeddings visualization, I use UMAP because it preserves both local and global structure better than t-SNE.

---

# Day 14 — Deep Learning & Transformers

## 1. Neural Networks from Scratch (Interview Level)

```
A neuron (perceptron):
  
  Input₁ ──w₁──┐
  Input₂ ──w₂──┤──Σ──→ activation(z) ──→ Output
  Input₃ ──w₃──┤
  Bias   ──────┘
  
  z = w₁x₁ + w₂x₂ + w₃x₃ + b
  output = activation(z)

A neural network = layers of neurons:
  
  Input Layer → Hidden Layer(s) → Output Layer
  [x₁, x₂, x₃]  [h₁, h₂, h₃, h₄]  [y₁, y₂]
       │              │                   │
    3 features    4 neurons            2 classes
```

### Activation Functions (Interview Must-Know!)

| Function | Formula | Range | When to Use | Interview Note |
|----------|---------|-------|-------------|---------------|
| **ReLU** ✅ | max(0, z) | [0, ∞) | Hidden layers (default) | "Solves vanishing gradient" |
| **Sigmoid** | 1/(1+e⁻ᶻ) | (0, 1) | Binary output | "Squishes to probability" |
| **Softmax** | eᶻⁱ/Σeᶻʲ | (0, 1) | Multi-class output | "Probabilities sum to 1" |
| **Tanh** | (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (-1, 1) | Hidden (legacy) | "Centered at 0, legacy" |
| **GELU** | z·Φ(z) | (-0.17, ∞) | Transformers | "Used in GPT, BERT" |
| **SiLU/Swish** | z·σ(z) | (-0.28, ∞) | Modern NNs | "Smooth ReLU, used in LLaMA" |

### Interview Q: "Why ReLU over Sigmoid in hidden layers?"
Sigmoid has the vanishing gradient problem — gradients approach 0 for large inputs, making deep networks untrainable. ReLU's gradient is either 0 or 1, so gradients flow cleanly through deep networks. The downside is 'dying ReLU' — neurons that output 0 forever. Leaky ReLU fixes this with a small slope for negative inputs.

---

## 2. Training Neural Networks — Backpropagation

```
Forward Pass:
  Input → multiply weights → activate → ... → prediction → loss

Backward Pass (backpropagation):
  Loss → compute gradients (chain rule) → update weights

The Chain Rule:
  ∂Loss/∂w = ∂Loss/∂output × ∂output/∂z × ∂z/∂w
  
  "How much does the loss change when I tweak this weight?"
  → computed layer by layer, from output back to input

Weight Update (Gradient Descent):
  w_new = w_old - learning_rate × gradient
  
  learning_rate = how big a step to take
    Too high → overshoots, diverges 💥
    Too low  → too slow, might get stuck
    Just right → converges to good solution ✅
```

### Optimizers (What Actually Updates Weights)

| Optimizer | Key Idea | When to Use |
|-----------|----------|-------------|
| **SGD** | Basic gradient descent with mini-batches | Simple, well-understood |
| **SGD + Momentum** | Accumulates past gradients → faster | Standard for vision |
| **Adam** ✅ | Adaptive learning rate per parameter | Default choice in 2026 |
| **AdamW** ✅ | Adam + proper weight decay | Used to train transformers/LLMs |
| **LAMB/LARS** | Layer-wise adaptive rates | Large-batch distributed training |

### Interview Q: "Explain gradient descent in plain English"
Imagine you're blindfolded on a mountain and need to reach the valley. You feel the slope under your feet (gradient) and take a step downhill (weight update). The learning rate is your step size — too big and you jump over the valley, too small and you take forever. Adam optimizer is like having adaptive shoes that take bigger steps on flat terrain and smaller steps near the valley.

### Resume Keyword: "Neural network training — backpropagation, optimization, regularization"

---

## 3. Loss Functions

```
Classification:
  ├── Binary Cross-Entropy: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  │   → Binary classification (spam/not spam)
  │
  └── Categorical Cross-Entropy: -Σ yᵢ·log(ŷᵢ)
      → Multi-class (cat/dog/bird)
      → THIS is what LLMs use! (predict next token from vocabulary)

Regression:
  ├── MSE (Mean Squared Error): mean((y - ŷ)²)
  │   → Penalizes large errors heavily (squared!)
  │
  ├── MAE (Mean Absolute Error): mean(|y - ŷ|)
  │   → More robust to outliers
  │
  └── Huber Loss: MSE when error small, MAE when error large
      → Best of both worlds

LLM Connection:
  Pre-training: Cross-entropy over vocabulary (next-token prediction)
  RLHF:         PPO (reinforcement learning loss) on preference data
```

### Interview Q: "MSE vs MAE?"
MSE penalizes large errors quadratically — one big mistake hurts a lot. MAE treats all errors linearly — more robust to outliers. I use MSE when large errors are unacceptable (financial forecasting), MAE when outliers are expected (user ratings). Huber loss combines both — MSE near 0, MAE far from 0.

---

## 4. Transformers — The Architecture Behind All LLMs

```
This is THE most important deep learning concept for GenAI interviews.

The Evolution:
  RNN (1986) → LSTM (1997) → Attention (2014) → Transformer (2017) → GPT/BERT/LLMs

Why Transformers replaced RNNs:
  RNN:         Process tokens one by one (sequential, slow)
  Transformer: Process ALL tokens at once (parallel, fast!)
```

### Self-Attention (The Core Innovation)

```
"Attention Is All You Need" (2017 paper, Google)

Input: "The cat sat on the mat"

For each word, attention asks:
  "How much should I focus on every other word?"

Attention("cat") might produce:
  "The" → 0.05  (low attention)
  "cat" → 0.30  (self)
  "sat" → 0.25  (what cat does)
  "on"  → 0.05  (low)
  "the" → 0.05  (low)
  "mat" → 0.30  (where cat is)

Math: Attention(Q, K, V) = softmax(QK^T / √d_k) × V

  Q (Query):  "What am I looking for?"
  K (Key):    "What do I contain?"
  V (Value):  "What information do I provide?"
  √d_k:      Scaling factor (prevents softmax saturation)

Multi-Head Attention:
  Run 8-96 attention heads in parallel
  Each head learns different relationships
  Head 1: syntax (subject-verb)
  Head 2: coreference (he → John)
  Head 3: semantic similarity
  Concat all heads → linear projection
```

### The Full Transformer Architecture

```
┌─────────────────────────────────────────────┐
│              TRANSFORMER                     │
│                                              │
│  ENCODER (BERT, embeddings)                  │
│  ┌────────────────────────────────────────┐  │
│  │  Input Embedding + Positional Encoding │  │
│  │       ↓                                │  │
│  │  Multi-Head Self-Attention             │  │
│  │       ↓                                │  │
│  │  Add & Normalize (residual connection) │  │
│  │       ↓                                │  │
│  │  Feed-Forward Network                  │  │
│  │       ↓                                │  │
│  │  Add & Normalize                       │  │
│  │       ↓                                │  │
│  │  (repeat N times)                      │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  DECODER (GPT, LLMs)                        │
│  ┌────────────────────────────────────────┐  │
│  │  Same as encoder PLUS:                 │  │
│  │  - Masked self-attention (can't see    │  │
│  │    future tokens — causal masking)     │  │
│  │  - Cross-attention (attend to encoder  │  │
│  │    output, for encoder-decoder models) │  │
│  └────────────────────────────────────────┘  │
│                                              │
└─────────────────────────────────────────────┘

Model types:
  Encoder-only:  BERT (bidirectional, good for understanding)
  Decoder-only:  GPT, LLaMA, Claude (autoregressive, good for generation)
  Encoder-Decoder: T5, BART (good for translation, summarization)

GPT = Decoder-only Transformer
  → Predicts next token, can't look ahead (masked attention)
  → This is why it's called "autoregressive"
```

### Interview Buzzword: "Self-Attention / Transformer Architecture"
Self-attention lets each token attend to every other token in the sequence, computing relevance scores as scaled dot-product of Query and Key vectors. It's O(n²) in sequence length, which is why context window limits exist. GPT uses masked (causal) self-attention so tokens can only attend to previous tokens — it generates left-to-right.

### Interview Q: "Why is attention O(n²) and why does it matter?"
Each token computes attention with every other token — that's n × n operations. For a 128K context window, that's 16 billion operations per layer. This is why there's active research on efficient attention: Flash Attention reduces memory from O(n²) to O(n) by tiling, and models like Mamba use state-space models for O(n) complexity.

---

## 5. Positional Encoding — How Transformers Know Word Order

```
Problem: Attention is permutation-invariant — it doesn't know
         "The dog bit the man" ≠ "The man bit the dog"

Solution: Add position information to each token embedding

Original (2017): Sinusoidal positional encoding
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Modern approaches:
  - RoPE (Rotary Positional Encoding): used in LLaMA, GPT-NeoX
    → encodes position as rotation in embedding space
    → naturally extends to longer sequences
  
  - ALiBi (Attention with Linear Biases): used in BLOOM
    → adds distance-based bias to attention scores
    → no learned parameters!
```

### Interview Buzzword: "RoPE / Positional Encoding"
RoPE encodes position by rotating embedding vectors — it naturally captures relative positions and can extrapolate to longer sequences than seen in training. This is how models like LLaMA support extended context windows. ALiBi is an alternative that uses linear attention bias — simpler but less flexible.

---

## 6. Key Deep Learning Concepts (Quick Reference)

### Batch Normalization vs Layer Normalization

| | Batch Norm | Layer Norm ✅ |
|---|----------|------------|
| Normalizes across | Batch dimension | Feature dimension |
| Used in | CNNs, vision | Transformers, LLMs |
| Depends on batch size | Yes (problematic) | No ✅ |
| Interview tip | "Vision models" | "All modern LLMs use LayerNorm" |

### Residual Connections (Skip Connections)
```
  x → [Layer] → output + x → next
       ↑_________________________↑
              skip connection

Why: Allows gradient to flow directly through deep networks
     Without this, 100+ layer networks can't train
     Every transformer block has residual connections
```

### Dropout
```
During training: randomly set neurons to 0 (e.g., 10% dropout)
During inference: keep all neurons, scale by (1 - dropout_rate)

Effect: forces redundancy, prevents co-adaptation of neurons
        → acts as ensemble of many sub-networks
```

### Resume Keyword: "Deep learning — transformers, attention mechanisms, neural network optimization"

---

## 7. CNNs & RNNs — Quick Overview (Enough for SDE2)

### CNNs (Convolutional Neural Networks)
```
Purpose: Process images (spatial data)
Key idea: Sliding filters detect local patterns

  Image → [Conv→ReLU→Pool] × N → Flatten → Dense → Output
  
  Conv: learns edge/texture/shape detectors
  Pool: downsamples, adds translation invariance
  
  Modern: Vision Transformers (ViT) now match or beat CNNs
  Still asked: "What's a convolution?" "What's pooling?"
```

### RNNs / LSTMs (Legacy but Still Asked)
```
Purpose: Process sequences (text, time series)
Key idea: Hidden state passes information across time steps

  x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → output
         ↑_______↑_______↑
       hidden state carries memory

Problem: Vanishing gradient — can't learn long-range dependencies
Fix: LSTM (Long Short-Term Memory) — gates control information flow
     - Forget gate: what to forget
     - Input gate: what to remember
     - Output gate: what to output

REPLACED BY: Transformers (parallel, no vanishing gradient)
Still relevant: Time-series forecasting, edge devices
```

### Interview Q: "Why did Transformers replace RNNs?"
Three reasons: (1) Parallelization — RNNs process sequentially (slow), Transformers process all tokens at once (GPU-friendly). (2) Long-range dependencies — RNNs forget context over long sequences despite LSTMs, while attention directly connects any two positions. (3) Scalability — Transformers scale better with more data and compute, leading to the LLM revolution.

---

## 8. Transfer Learning & Fine-Tuning (Connects to GenAI!)

```
Transfer Learning: Use a model trained on Task A for Task B

Pre-training → Fine-tuning pipeline:
  1. Pre-train on massive data (expensive, done by OpenAI/Google)
     GPT: trained on internet text (next-token prediction)
     BERT: trained on internet text (masked token prediction)
  
  2. Fine-tune on your specific task (cheap, done by you)
     Option A: Full fine-tuning (update all parameters)
     Option B: LoRA (update ~1% of parameters) ← modern approach
     Option C: Prompt tuning (no weight changes, just better prompts)

This is EXACTLY how the GenAI stack works:
  Pre-trained LLM → Fine-tuned with RLHF → Your prompts/RAG
  (Days 13-14)      (Day 12)                (Days 1-6)
```

### Interview Buzzword: "Transfer Learning / LoRA"
Transfer learning reuses knowledge from pre-trained models — a model that learned language patterns from the entire internet can be fine-tuned for your specific task with just thousands of examples. LoRA makes this efficient by only training low-rank adapter matrices (~1% of parameters), keeping the base model frozen.

---

# Day 15 — ML Metrics, MLOps & Production ML

## 1. Classification Metrics

### The Confusion Matrix
```
                    PREDICTED
                  Positive  Negative
ACTUAL  Positive    TP        FN
        Negative    FP        TN

TP = True Positive  (correctly predicted positive)
FP = False Positive (incorrectly predicted positive) ← "Type I Error"
FN = False Negative (incorrectly predicted negative) ← "Type II Error"
TN = True Negative  (correctly predicted negative)
```

### Key Metrics

| Metric | Formula | What It Means | When to Prioritize |
|--------|---------|--------------|-------------------|
| **Accuracy** | (TP+TN) / Total | % correct overall | Balanced classes only! |
| **Precision** | TP / (TP+FP) | "Of predicted positive, how many are correct?" | Spam filter (don't misclassify good email) |
| **Recall** | TP / (TP+FN) | "Of actual positive, how many did we catch?" | Cancer detection (don't miss sick patients) |
| **F1 Score** | 2 × P×R / (P+R) | Harmonic mean of precision & recall | Balance between P and R |
| **ROC-AUC** | Area under ROC curve | Overall model discriminative power | Model comparison |

### Interview Buzzword: "Precision-Recall Tradeoff"
It's a tradeoff — increasing recall (catch more positives) often decreases precision (more false alarms). The business decides the balance: for fraud detection, high recall is critical (don't miss fraud even if some legit transactions get flagged). For email spam, high precision is critical (never put a real email in spam).

### Interview Q: "When is accuracy misleading?"
With imbalanced classes! If 99% of emails are not spam, a model that predicts 'not spam' for everything gets 99% accuracy but catches zero spam. I'd use F1 score, precision-recall AUC, or weighted metrics instead. I'd also consider class rebalancing: SMOTE for oversampling, class weights, or focal loss.

### Class Imbalance — How to Handle It
```
Problem: 99% negative, 1% positive (fraud, disease, defects)

Solutions:
  ├── Data-level
  │     ├── Oversampling minority: SMOTE (creates synthetic examples)
  │     ├── Undersampling majority: random or informed
  │     └── Augmentation: create more positive examples
  │
  ├── Algorithm-level
  │     ├── Class weights: penalize misclassifying minority more
  │     ├── Focal loss: down-weights easy examples, focuses on hard ones
  │     └── Anomaly detection: treat minority as anomalies
  │
  └── Evaluation-level
        ├── Use Precision-Recall AUC (not ROC-AUC)
        ├── Use F1 or F2 score
        └── Stratified cross-validation
```

### Resume Keyword: "Classification metrics — precision, recall, F1, AUC, class imbalance handling"

---

## 2. Regression Metrics

| Metric | Formula | Interpretation | Interview Note |
|--------|---------|---------------|---------------|
| **MSE** | mean((y-ŷ)²) | Average squared error | Penalizes large errors |
| **RMSE** | √MSE | Same units as target | Most interpretable |
| **MAE** | mean(\|y-ŷ\|) | Average absolute error | Robust to outliers |
| **R²** | 1 - SS_res/SS_tot | % variance explained | 0.85 = "explains 85%" |
| **MAPE** | mean(\|y-ŷ\|/y) × 100 | % error | Business-friendly |

---

## 3. MLOps — The Production ML Stack

```
MLOps = DevOps for Machine Learning

Your LLMOps knowledge (Days 6, 11) IS MLOps applied to LLMs!

            TRADITIONAL MLOps          YOUR LLMOps
            ─────────────────          ────────────
Data:       Feature store              Embedding store (pgvector)
Training:   Training pipeline          Fine-tuning / Prompt Engineering
Serving:    Model server (TorchServe)  API call (OpenAI, Anthropic)
Monitoring: Data drift, model decay    Token cost, prompt quality
Eval:       Offline metrics (F1, AUC)  LLM evals (faithfulness, RAGAS)
Registry:   MLflow model registry      Prompt registry (Langfuse)
CI/CD:      Train → test → deploy      Prompt → eval → deploy
```

### The MLOps Maturity Model

| Level | Description | Your Status |
|-------|-------------|-------------|
| **L0** | Manual, notebook-based ML | ❌ Don't be here |
| **L1** | Automated training pipelines | SDE2 baseline |
| **L2** | Automated CI/CD for ML, monitoring | SDE2 target |
| **L3** | Automated retraining on data drift | Senior/Staff |

### Key MLOps Tools (Interview Awareness)

| Category | Tool | Notes |
|----------|------|----------------------|
| **Experiment Tracking** | MLflow, W&B | "Track hyperparams, metrics, artifacts" |
| **Feature Store** | Feast, Tecton | "Consistent features in training & serving" |
| **Model Serving** | TorchServe, Triton, BentoML | "Low-latency model inference APIs" |
| **Pipeline** | Kubeflow, Airflow, Prefect | "Orchestrate training & data pipelines" |
| **Model Registry** | MLflow, Vertex AI | "Version, stage, and deploy models" |
| **Monitoring** | Arize, WhyLabs, Evidently | "Detect data drift & model decay" |

### Interview Buzzword: "Data Drift / Model Decay"
Data drift is when production data distribution shifts from training data — a model trained on 2024 user behavior might fail in 2026. Concept drift is when the underlying relationship changes. I monitor with statistical tests (KS test, PSI) and retrain on a schedule or trigger-based.

### Resume Keyword: "MLOps — experiment tracking, model serving, drift monitoring, CI/CD for ML"

---

## 4. Model Deployment Patterns

```
Pattern 1: BATCH PREDICTION
  ┌──────────┐    ┌───────┐    ┌──────────┐
  │ New Data  │───→│ Model │───→│ Results  │ → store in DB
  └──────────┘    └───────┘    └──────────┘
  Schedule: nightly, hourly
  Use case: recommendation systems, risk scoring
  Your GenAI version: Batch API (50% cheaper!)

Pattern 2: REAL-TIME INFERENCE (API)
  ┌──────┐    ┌───────────┐    ┌───────┐
  │ User │───→│ API Server │───→│ Model │───→ response
  └──────┘    └───────────┘    └───────┘
  Latency: < 100ms
  Use case: fraud detection, search, chatbot
  Your GenAI version: OpenAI API calls

Pattern 3: EDGE DEPLOYMENT
  Model runs on user's device (phone, browser)
  Pros: No network latency, privacy
  Cons: Limited compute, model size constraints
  Tools: ONNX, TensorFlow Lite, CoreML
  Your GenAI version: Small models (GPT-5 nano, Gemma)

Pattern 4: SHADOW DEPLOYMENT (A/B for ML)
  ┌──────┐    ┌────────────┐    ┌─────────────┐
  │ User │───→│ Current     │───→│ Response     │ ← served
  └──────┘    │ Model       │    └─────────────┘
              │             │
              │ Shadow      │───→ logged, compared
              │ Model (new) │    (not served to user)
              └────────────┘
  
  Your GenAI version: A/B test prompts (Day 11!)
```

---

## 5. Connecting ML Foundations to Your GenAI Stack

```
THIS IS YOUR INTERVIEW SUPERPOWER — connecting classical ML to GenAI

Classical ML concept         → GenAI equivalent (YOUR knowledge)
────────────────────         ─────────────────────────────────
Feature engineering          → Prompt engineering (Day 2)
TF-IDF text features         → Embeddings (Day 4)
KNN search                   → Vector search / RAG (Day 5)
Cross-validation             → LLM evaluation / RAGAS (Day 11)
Model selection              → Model routing — nano/mini/full (Day 6)
Ensemble methods             → Multi-agent systems (Day 9)
Regularization               → Guardrails (Day 10)
Transfer learning            → Fine-tuning + LoRA (Day 12)
Model monitoring             → LLMOps / Langfuse (Day 6)
A/B testing                  → Prompt A/B testing (Day 11)
Data pipeline                → RAG ingestion pipeline (Day 5)
Batch prediction             → Batch API / Flex tier (Day 6)
Feature store                → Vector database (Day 4)
Loss function (cross-entropy)→ Next-token prediction (how GPT trains!)
SGD / Adam optimizer          → How LLMs are actually trained
Data drift                   → Embedding drift / prompt decay
```

### Interview Superpower Answer
The GenAI stack is built on classical ML foundations. Embeddings are learned feature representations. RAG is KNN search on those features. Prompt engineering is the new feature engineering. LLM evaluation is cross-validation for generative models. When I debug a RAG pipeline, I apply the same diagnostic thinking as debugging any ML system: check data quality, check feature quality (embeddings), check model selection, check evaluation metrics.

---

## 6. Math You Should Know (Just Enough)

### Probability (30-second refresher)
```
Bayes' Theorem:
  P(A|B) = P(B|A) × P(A) / P(B)
  
  "Probability of spam GIVEN these words"
  = P(these words | spam) × P(spam) / P(these words)

Key concepts:
  - Conditional probability: P(A|B) — A given B
  - Independence: P(A,B) = P(A) × P(B)
  - Expected value: E[X] = Σ xᵢ × P(xᵢ)
  - Variance: how spread out values are
  - Normal distribution: bell curve (mean, std dev)
```

### Linear Algebra (The Embedding Connection!)
```
Vectors: Your embeddings ARE vectors! [0.1, -0.3, 0.5, ...]
  - Dot product: a·b = Σ aᵢbᵢ → similarity measure
  - Cosine similarity: dot product / (||a|| × ||b||) → YOUR Day 4!
  - Matrix multiplication: how attention works (Q × K^T)

Matrices: Neural network weights ARE matrices!
  Output = Activation(Weight_matrix × Input + Bias)
  
  That's literally what every layer does:
  [h₁]   [w₁₁ w₁₂ w₁₃]   [x₁]   [b₁]
  [h₂] = [w₂₁ w₂₂ w₂₃] × [x₂] + [b₂]
  [h₃]   [w₃₁ w₃₂ w₃₃]   [x₃]   [b₃]
```

### Gradient Descent (The Learning Algorithm)
```
Goal: Find weights that minimize the loss function

  w = w - α × ∂L/∂w
  
  α = learning rate (step size)
  ∂L/∂w = gradient (direction of steepest ascent, we go opposite)

Variants:
  - Batch GD: use ALL data per update (slow, smooth)
  - Stochastic GD (SGD): use 1 sample per update (noisy, fast)
  - Mini-batch GD: use 32-512 samples (practical sweet spot)
  
Mini-batch is what everyone uses. "SGD" in practice means mini-batch.
```

---

# Interview Cheat Sheet — ML Buzzwords

## Resume Section: "ML Foundations"

### For SDE2 + AI Engineer
```
Classical ML: Logistic Regression, Random Forest, XGBoost/LightGBM,
              SVM, K-Means, PCA, Feature Engineering, Cross-Validation

Deep Learning: Neural Networks, Backpropagation, Adam Optimizer,
               CNNs (vision), RNNs/LSTMs (legacy), Dropout, BatchNorm

Transformers: Self-Attention, Multi-Head Attention, Positional
              Encoding (RoPE), Encoder-Decoder Architecture,
              Transfer Learning, LoRA Fine-Tuning

Evaluation: Precision/Recall/F1/AUC, Confusion Matrix, Class
            Imbalance (SMOTE), Cross-Validation, A/B Testing

MLOps: Experiment Tracking (MLflow, W&B), Model Serving,
       Data Drift Monitoring, CI/CD for ML Pipelines
```

---

## Top 25 ML Interview Buzzwords

### Tier 1 — Asked in EVERY ML-adjacent interview
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 1 | **Bias-Variance Tradeoff** | Underfitting vs overfitting balance |
| 2 | **Overfitting / Regularization** | Model memorizes noise; fix with L1/L2/dropout |
| 3 | **Gradient Descent** | Iterative optimization by following gradients downhill |
| 4 | **Precision vs Recall** | Correctness vs completeness tradeoff |
| 5 | **Cross-Validation** | K-fold evaluation to prevent over-optimistic metrics |
| 6 | **Transformer / Self-Attention** | Architecture that processes all tokens in parallel |
| 7 | **Feature Engineering** | Transforming raw data into useful model inputs |
| 8 | **Supervised vs Unsupervised** | Labeled data (predict) vs unlabeled (discover) |

### Tier 2 — Asked in 50%+ of interviews
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 9 | **XGBoost / Gradient Boosting** | Sequential ensemble — each tree fixes previous errors |
| 10 | **Random Forest / Bagging** | Parallel ensemble — many trees vote for answer |
| 11 | **Transfer Learning** | Reuse pre-trained model for new task |
| 12 | **Backpropagation** | Chain rule to compute gradients layer by layer |
| 13 | **Data Drift** | Production data shifts from training distribution |
| 14 | **Class Imbalance** | Unequal class distribution → misleading accuracy |
| 15 | **Activation Functions (ReLU)** | Non-linearity that enables deep learning |
| 16 | **Loss Function** | Math that defines "how wrong is the model" |

### Tier 3 — Differentiators (AI Engineer level)
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 17 | **LoRA / Parameter-Efficient Fine-Tuning** | Train 1% of parameters, freeze the rest |
| 18 | **Attention (Q, K, V)** | Query-Key-Value mechanism for token relationships |
| 19 | **RoPE / Positional Encoding** | How transformers know word order |
| 20 | **Flash Attention** | Memory-efficient attention (O(n) memory vs O(n²)) |
| 21 | **Data Leakage** | Information from test set contaminates training |
| 22 | **KV Cache** | Cache past key-value pairs for faster autoregressive generation |
| 23 | **Quantization** | Reduce model precision (FP32 → INT4) for speed/memory |
| 24 | **RLHF** | Human preferences to align LLM behavior |
| 25 | **Mixture of Experts (MoE)** | Route tokens to specialized sub-networks |

---

# ML System Design Patterns

## Interview Q: "Design a fraud detection system"

```
┌──────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION SYSTEM                      │
│                                                               │
│  DATA PIPELINE                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Transaction stream → Feature engineering:              │ │
│  │    - Amount, merchant, time, location                   │ │
│  │    - Velocity features (txns in last 1h/24h/7d)        │ │
│  │    - User behavior deviation (avg amount, freq)         │ │
│  │    - Merchant risk score                                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  MODEL LAYER                                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Level 1: Rules engine (hard limits, known patterns)    │ │
│  │       ↓                                                  │ │
│  │  Level 2: XGBoost classifier (tabular features)         │ │
│  │       ↓                                                  │ │
│  │  Level 3: LLM analysis for edge cases (GenAI!)          │ │
│  │           "Analyze this transaction pattern and          │ │
│  │            explain if it looks fraudulent"               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  EVALUATION & OPS                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Metrics: Precision (don't block legit users)           │ │
│  │           Recall (don't miss fraud)                     │ │
│  │  Class imbalance: 99.9% legit, 0.1% fraud → SMOTE      │ │
│  │  Latency: < 100ms (real-time decision)                  │ │
│  │  Monitoring: data drift on transaction patterns         │ │
│  │  Human-in-loop: flag uncertain cases for review         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Key talking points:
1. **Class imbalance** is the #1 challenge — 0.1% fraud rate
2. **Feature engineering** matters more than model choice for tabular data
3. **Latency requirements** dictate architecture — can't use a slow model
4. **XGBoost > neural nets** for tabular data (faster, more interpretable)
5. **GenAI integration**: LLM for explaining decisions, analyzing edge cases
6. **Monitoring**: fraud patterns evolve → need drift detection & retraining

---

## Interview Q: "How does ML connect to your GenAI work?"

```
┌─────────────────────────────────────────────────────────────┐
│              YOUR COMPLETE INTERVIEW TOOLKIT                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FOUNDATION (Day 13)                                  │   │
│  │  Classical ML: LogReg, XGBoost, Random Forest, SVM    │   │
│  │  Concepts: Bias-Variance, Regularization, CV          │   │
│  │  "I understand WHY models work"                       │   │
│  └──────────────────────────────────────────────────────┘   │
│           ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  DEEP LEARNING (Day 14)                               │   │
│  │  Neural Nets, Transformers, Attention, Backprop       │   │
│  │  "I understand HOW LLMs work under the hood"          │   │
│  └──────────────────────────────────────────────────────┘   │
│           ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GENAI APPLICATION (Days 1-12)                        │   │
│  │  APIs, Prompting, RAG, Agents, MCP, Eval, Prod        │   │
│  │  "I BUILD production AI systems"                      │   │
│  └──────────────────────────────────────────────────────┘   │
│           ↓                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PRODUCTION ML (Day 15)                               │   │
│  │  Metrics, MLOps, Deployment, Monitoring               │   │
│  │  "I OPERATE AI systems at scale"                      │   │
│  └──────────────────────────────────────────────────────┘   │
│           +                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  SDE2 FUNDAMENTALS (separate prep)                    │   │
│  │  DSA (LeetCode) + System Design (Grokking)            │   │
│  │  "I'm a strong engineer FIRST"                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  = COMPLETE SDE2 + AI ENGINEER CANDIDATE ✅                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

# Glossary

| Term | Plain English Definition |
|------|------------------------|
| **Activation function** | Non-linearity applied after each layer (ReLU, sigmoid) |
| **Adam** | Adaptive optimizer — default choice for training neural nets |
| **AUC-ROC** | Area under the ROC curve — overall classification performance |
| **Backpropagation** | Algorithm to compute gradients using the chain rule |
| **Bagging** | Train multiple models on random data subsets, average results |
| **Batch normalization** | Normalize layer inputs across the batch (vision models) |
| **Bias-variance tradeoff** | Simple model (high bias) vs complex model (high variance) |
| **Boosting** | Sequentially train models, each fixing previous errors |
| **Class imbalance** | Unequal class distribution (99% vs 1%) |
| **Confusion matrix** | Table of TP, FP, TN, FN |
| **Cross-entropy** | Loss function for classification (and LLM pre-training!) |
| **Cross-validation** | K-fold evaluation for unbiased performance estimates |
| **Data drift** | Production data distribution shifts from training data |
| **Data leakage** | Test set info contaminates training → inflated metrics |
| **Decision tree** | Nested if-else conditions for classification/regression |
| **Dropout** | Randomly disable neurons during training to prevent overfitting |
| **Ensemble** | Combine multiple models for better predictions |
| **F1 score** | Harmonic mean of precision and recall |
| **Feature engineering** | Transforming raw data into useful model inputs |
| **Fine-tuning** | Adapting a pre-trained model to a specific task |
| **Flash Attention** | Memory-efficient attention algorithm (O(n) memory) |
| **Gradient descent** | Optimize weights by stepping in direction of negative gradient |
| **KV Cache** | Store past key-value pairs to speed up autoregressive generation |
| **L1/L2 regularization** | Penalties on weight magnitude to prevent overfitting |
| **Layer normalization** | Normalize across features (used in all transformers) |
| **Learning rate** | Step size for gradient descent updates |
| **LoRA** | Low-rank adapter — efficient fine-tuning (~1% parameters) |
| **Loss function** | Mathematical objective that defines model error |
| **LSTM** | Gated RNN variant that handles long-range dependencies |
| **Mini-batch** | Process subset of data per gradient update (32-512 samples) |
| **MLOps** | DevOps practices applied to ML model lifecycle |
| **MoE** | Mixture of Experts — route inputs to specialized sub-networks |
| **Overfitting** | Model memorizes training data, fails on new data |
| **PCA** | Dimensionality reduction via principal components |
| **Precision** | Of predicted positives, how many are truly positive |
| **Quantization** | Reduce numerical precision (FP32→INT8) for speed/memory |
| **Random Forest** | Ensemble of decision trees trained on random subsets |
| **Recall** | Of actual positives, how many were correctly identified |
| **Residual connection** | Skip connection that adds input directly to layer output |
| **RLHF** | Reinforcement Learning from Human Feedback — aligns LLMs |
| **RoPE** | Rotary Positional Encoding — how modern LLMs encode position |
| **Self-attention** | Each token attends to all other tokens (Q×K^T×V) |
| **Self-supervised** | Model creates its own labels (next-token prediction for LLMs) |
| **SMOTE** | Synthetic Minority Oversampling for class imbalance |
| **Softmax** | Converts scores to probabilities that sum to 1 |
| **SVM** | Finds maximum-margin hyperplane between classes |
| **Transfer learning** | Reuse pre-trained model knowledge for new tasks |
| **Transformer** | Attention-based architecture behind all modern LLMs |
| **Underfitting** | Model is too simple to capture data patterns |
| **Vanishing gradient** | Gradients shrink to ~0 in deep networks (fixed by ReLU) |
| **XGBoost** | Gradient boosting library — king of tabular data |

---

## 🗺️ Complete Study Guide Map

```
YOUR COMPLETE INTERVIEW PREP:

  ┌─ DSA / LeetCode ──────────────── (separate prep) ──────┐
  ├─ System Design (Grokking) ─────── (separate prep) ──────┤
  │                                                          │
  ├─ Days 1-3:  APIs + Prompting + Function Calling ────────┤
  ├─ Days 4-6:  Embeddings + RAG + Observability ───────────┤
  ├─ Days 7-12: Agents + MCP + Multi-Agent + Eval + Prod ──┤
  ├─ Days 13-15: ML Foundations + DL + Transformers + MLOps ┤
  │                                                          │
  └─ = COMPLETE SDE2 + AI/GenAI ENGINEER CANDIDATE ─────────┘

  Total: ~4,000 lines of interview-ready knowledge
         90 buzzwords ranked by interview frequency
         4 system design patterns with architecture diagrams
         4 copy-paste resume sections
```

---

> 💡 **Pro Tip:** In interviews, always connect ML fundamentals to your GenAI work. Don't just say "I know XGBoost" — say "I use XGBoost for tabular data and LLMs for unstructured data. My RAG pipeline uses the same evaluation mindset as traditional ML — precision and recall, just applied to retrieval quality."

> **Study order:** If short on time, prioritize Tier 1 buzzwords first. For SDE2 + GenAI roles, interviewers care more about "can you build and ship AI systems" (Days 1-12) than "can you derive backpropagation" — but they DO want to see ML literacy (this guide).

---

*Part of the complete 15-day interview prep series. Pairs with Days 1-3 (APIs), Days 4-6 (RAG), and Days 7-12 (Agents & Production GenAI).*
