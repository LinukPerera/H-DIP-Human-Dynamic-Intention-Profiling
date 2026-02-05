# H-DIP: Human Dynamic Intention Profiling

![H-DIP Framework](https://github.com/LinukPerera/H-DIP-Human-Dynamic-Intention-Profiling/blob/main/Images/Final_Tx_FMW.jpeg)

**H-DIP (Human Dynamic Intention Profiling)** is a generalized kinematic framework designed to understand and predict human intent through motion. Originally developed for high-precision hand gesture recognition, the framework has evolved into a universal motion analysis engine capable of distinguishing complex behavioral nuances—from the micro-movements of a finger to the gross motor dynamics of gait and dance.

By integrating **Optimized Average Weighted Acceleration (OAWA)** profiling with a custom **Spatio-Temporal Transformer**, H-DIP achieves robust generalization, successfully interpreting unseen motion classes (such as dancing) by understanding the fundamental physics of stability and intent.

---

## 1. The Roots: Hand Gesture Recognition
*From Stochastic Motion to Deterministic Intent*

The foundation of H-DIP lies in solving the "Recoil Problem" in hand gesture recognition. Traditional models often confused the "return to center" (recoil) of a hand with an active gesture. To solve this, we developed a mathematical profiling approach that isolates intentional force from passive momentum.

### The Architecture (Phase I)
Our initial approach utilized a hybrid pipeline combining **Convolutional Neural Networks (CNNs)** for spatial feature extraction, **Dense Neural Networks (DNN)** for classification, and **LSTMs** for temporal sequence modeling.

![Original LSTM DNN Framework](https://github.com/LinukPerera/H-DIP-Human-Dynamic-Intention-Profiling/blob/main/Images/Org_LSTM_DNN_FRMW.jpeg)

### Core Innovation: The OAWA Algorithm
The **Optimized Average Weighted Acceleration (OAWA)** algorithm is the mathematical heart of this phase. It acts as a high-pass filter for "intention," favoring the rapid acceleration changes typical of voluntary movement while damping the smooth deceleration of recoils.

**Mathematical Formulation:**

1.  **Keypoint Weighting ($w_i$):**
    We assign weights based on a binomial-derived linear distribution, heavily favoring fingertips ($i \in \{4,8,12,16,20\}$) where kinetic energy is highest during gestures.

    where $w_i = 6$ for fingertip keypoints ($i \in \{4,8,12,16,20\}$), and $w_i = 1$ otherwise.


2.  **Weighted Velocity Profiling ($V_{avg}$):**
    We compute the weighted sum of velocities across all keypoints to create a global motion profile.

    $$V_{\text{avg, axis}}(t) = \sum_{i=0}^{20} v_{i,\text{axis}}(t) \cdot w_i$$

3.  **Acceleration Profiling (OAWA Output):**
    The derivative of the velocity profile yields the acceleration profile, which highlights the *impulse* of the gesture.
    $$A_{\text{avg, axis}}(t) = \sum_{i=0}^{20} \left( \frac{v_{i,\text{axis}}(t) - v_{i,\text{axis}}(t_{\text{prev}})}{t - t_{\text{prev}}} \right) \cdot w_i$$

This approach allowed us to virtually "understand" every permutation of a gesture (e.g., a swipe performed by one finger vs. whole hand) using a very small training set.

### Results: Clustering Efficiency
The effectiveness of the OAWA weighting strategy is visually demonstrable when analyzing feature clustering. By assigning higher weights to high-energy keypoints (fingertips), we significantly reduce noise and improve the separability of gesture classes.

**1. Raw Average Velocity (Unweighted)**
In the unweighted scatter plot, gesture classes overlap significantly, making it difficult for the model to draw clear decision boundaries. The data appears noisy and less distinct.

![Average Velocity Scatter Plot](https://github.com/LinukPerera/Neural-Network-Driven-Augmented-Reality-for-Gesture-Control/blob/main/Project-Results/Average%20Velocity%20Scatter%20Plot.png)

**2. Weighted Average Velocity (OAWA Applied)**
After applying the OAWA weighting scheme, the features for each class become tightly clustered and distinct. The "Swipes" and "Static" gestures separate clearly, proving that the algorithm successfully amplifies the signal (intent) while suppressing the noise.

![Average Velocity Weighted](https://github.com/LinukPerera/Neural-Network-Driven-Augmented-Reality-for-Gesture-Control/blob/main/Project-Results/Average%20Velocity%20Weighted.png)

### Phase I Results
The initial models demonstrated significant success in isolating intent:
* **Method 2 (Neural Network):** Achieved **97% accuracy** on independent datasets.
* **Method 3 (OAWA + LSTM):** Achieved **88% accuracy**, with superior performance in dynamic recoil filtering.

*(See legacy documentation for full confusion matrices and performance reports.)*

---

## 2. The Evolution: Transformer Architecture
*Moving Beyond Recurrent Networks*

While LSTMs handled temporal data well, they struggled with long-term dependencies and parallelization. To scale the OAWA features to complex, multi-modal human actions, we transitioned to a **Transformer-based architecture**.

![Transformer Architecture](https://github.com/LinukPerera/H-DIP-Human-Dynamic-Intention-Profiling/blob/main/Images/Tx_Arc.jpeg)

**Key Architectural Features:**
* **Input Embedding:** The OAWA acceleration profiles are projected into a higher-dimensional space ($d_{model} = 128$).
* **Positional Encoding:** Standard sinusoidal encodings are added to retain the temporal order of the motion sequence.
* **Multi-Head Attention:** The core mechanism allowing the model to focus on specific "moments of intent" (e.g., the peak acceleration of a swipe or the heel-strike of a step) regardless of their position in the time series.
* **CLS Token:** A learnable classification token aggregates the global context of the motion sequence for the final MLP head.

---

## 3. H-DIP: The Unified Framework
*Generalizing to Human Behavior*

The **H-DIP Framework** (top image) unifies the Spatio-Temporal Graph Convolutional Networks (ST-GCN) for skeletal extraction, the OAWA algorithm for intention filtering, and the Transformer for global reasoning.

This architecture is not limited to hands; it generalizes to any articulated system. By analyzing the *physics* of movement (acceleration, stability, symmetry) rather than just the *image* of movement, H-DIP can infer the state of the subject.

---

## 4. Case Study: Generalization & The "Dancing" Anomaly
*Proof of Concept via Fall Detection*

To test the limits of H-DIP's generalization, we retrained the architecture for **Fall Detection and Gait Analysis**. The goal was to distinguish between "Steady" and "Unsteady" (fall-risk) individuals.

### The Data
The model was trained strictly on standard walking patterns and simulated unsteady gaits.

| **Unsteady Gait (Fall Risk)** | **Steady Gait (Normal Walking)** |
| :---: | :---: |
| ![Unsteady Gait](https://github.com/LinukPerera/H-DIP-Human-Dynamic-Intention-Profiling/blob/main/Images/Gait.gif) | ![Steady Gait](https://github.com/LinukPerera/H-DIP-Human-Dynamic-Intention-Profiling/blob/main/Images/Wlk_Trx.gif) |
| *High variance in acceleration profiles.* | *Rhythmic, consistent acceleration profiles.* |

### The "Dancing" Anomaly (Surprising Result)
During inference testing, we fed the model a video of a person dancing—a class of motion it had **never seen** during training.

A traditional model might classify this as "Unsteady" due to the erratic, high-speed movements and non-linear trajectories. However, H-DIP successfully recognized the subject as **Steady**.

| **The "Dancing" Anomaly** |
| :---: |
| ![Dancing Inference](https://github.com/LinukPerera/H-DIP-Human-Dynamic-Intention-Profiling/blob/main/Images/TX_Dancing.gif) |
| *Result: **Steady (Safe)**. The model understood that despite the complexity, the motion was controlled and intentional.* |

**Scientific Conclusion:**
This result confirms that H-DIP does not merely memorize visual patterns. By profiling the *dynamics* of the movement (via OAWA) and the *context* (via Transformers), it correctly deduced that the dancer's center of mass and acceleration derivatives indicated **controlled stability**, distinguishing it from the chaotic instability of a fall.

---

## Installation & Usage

### Prerequisites
* Python 3.9+
* TensorFlow / PyTorch
* MediaPipe
* OpenCV

### Quick Start
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/LinukPerera/H-DIP-Human-Dynamic-Intention-Profiling.git](https://github.com/LinukPerera/H-DIP-Human-Dynamic-Intention-Profiling.git)
    cd H-DIP
    ```

2.  **Collect Data (Generic)**
    Use the generalized collection script to capture velocity/acceleration profiles for any articulated motion.
    ```bash
    python DataSet_collect.py
    ```

3.  **Train the Transformer**
    ```bash
    python train_transformer.py
    ```

4.  **Live Inference**
    ```bash
    python live_inference.py
    ```

## Contributing
H-DIP is an open research project. We welcome contributions, especially in applying the framework to new domains (sports analytics, rehabilitation monitoring, robotic teleoperation).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
