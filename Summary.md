Research Paper 1: Homomorphic Encryption for Machine Learning Applications with CKKS Algorithms: A Survey of Developments and Applications

This paper, Homomorphic Encryption for Machine Learning Applications with CKKS Algorithms: A Survey of Developments and Applications, reviews the integration of Homomorphic Encryption (HE) into Privacy-Preserving Machine Learning (PPML). It focuses specifically on the advantages of the Cheon-Kim-Kim-Song (CKKS) algorithm [cite: 1.1.10]. The paper highlights how CKKS is uniquely suited for machine learning because it supports approximate floating-point computations, making it possible to run algorithms like K-nearest neighbors (KNN), K-means clustering, and face recognition on encrypted data [cite: 1.1.10]. However, the authors note that integrating HE and ML creates significant bottlenecks: encrypted computations can extend processing times from hours to days, and cryptographic noise can reduce the final model's accuracy [cite: 1.1.10].

Important Key Points, Algorithms, and Optimizations
The CKKS Algorithm: This is the core algorithm discussed because its ability to encrypt floating-point numbers bridges the gap between strict cryptographic security and the real-number arithmetic inherently required by ML models [cite: 1.1.10].

Fog Computing (Edge-Cloud Collaboration): Traditional cloud architectures face severe latency bottlenecks when handling real-time encrypted data (like medical diagnoses) [cite: 1.1.10]. The paper proposes deploying "intelligent fog nodes" at the network edge to handle data storage and perform partial encrypted computations close to the data source, drastically reducing communication delays [cite: 1.1.10].

Algebraic Optimization & Noise Management: To make CKKS practical, the paper emphasizes combining algebraic optimizations (to improve computational speed) with systematic noise management (to ensure the error introduced by approximate encryption does not destroy the model's accuracy) [cite: 1.1.10].

Implementation Strategies for MIMIC-III and MIMIC-IV Datasets
When scaling these concepts to massive, highly complex Electronic Health Record (EHR) datasets like MIMIC-III and MIMIC-IV, you should leverage the paper's insights in the following ways:

Utilize CKKS for Continuous Clinical Variables: MIMIC datasets are heavily reliant on continuous physiological data (e.g., heart rate, blood pressure, lab values like creatinine or bilirubin). Because CKKS supports floating-point approximation, you can encrypt these normalized continuous variables directly without rounding them to integers. This preserves the exact statistical precision needed for accurate clinical risk scoring.

Deploy Fog Computing Architectures for ICUs: MIMIC data represents intensive care unit (ICU) telemetry. Transmitting massive, continuous streams of encrypted MIMIC data to a central cloud server will cause unacceptable latency. You should emulate the paper's "fog computing" approach by treating individual ICUs or hospital local servers as edge nodes [cite: 1.1.10]. These nodes can aggregate, batch, and partially compute encrypted patient states locally before sending them to a central model, mimicking a real-world, low-latency hospital deployment.

Optimize KNN for Patient Similarity: The paper discusses integrating CKKS with K-Nearest Neighbors (KNN) [cite: 1.1.10]. You can apply this to MIMIC by building an encrypted KNN model to find "similar patient profiles" based on historical records. However, because computing encrypted distances across millions of MIMIC rows is computationally expensive, you must heavily rely on algebraic optimizations—such as SIMD (Single Instruction, Multiple Data) packing—to encode multiple patient vectors into a single ciphertext array to speed up distance calculations.

Manage Noise in Deep Predictive Models: If you are building complex predictive models on MIMIC (like predicting mortality or sepsis onset), the multiplicative depth (the number of consecutive multiplications in the algorithm) will be high. Because CKKS is an approximate scheme, each multiplication adds noise. You must implement strict systematic noise management (such as periodic bootstrapping or ciphertext rescaling) to ensure the accumulated noise does not flip a binary clinical prediction (e.g., accidentally changing a "high risk" output to "low risk" due to encryption artifacts).


Research Paper 2:A comprehensive survey on secure healthcare data processing
with homomorphic encryption: attacks and defenses

This comprehensive survey by Lee, Lim, and Eswaran (2025) systematically reviews the theoretical foundations, implementation schemes, and healthcare applications of Homomorphic Encryption (HE). Unlike other papers that solely focus on performance, this survey uniquely bridges the gap between HE's privacy-preserving capabilities (for EHRs, medical imaging, and machine learning) and its systemic vulnerabilities, detailing attack vectors like side-channel, chosen ciphertext, and fault injection attacks alongside their mitigation strategies.

Important Key Points
Four Tiers of HE: The paper categorizes HE into Partially Homomorphic Encryption (PHE), Somewhat Homomorphic Encryption (SHE), Fully Homomorphic Encryption (FHE), and Fully Leveled Homomorphic Encryption (FLHE), mapping each to specific computational constraints.

The Rise of FLHE: Fully Leveled Homomorphic Encryption is specifically highlighted as a tailored solution for deep learning. It manages noise growth for a predetermined number of operations, matching the layered structure of neural networks without the prohibitive computational overhead of pure FHE.

Vulnerability Awareness: The authors stress that mathematical encryption is not a silver bullet. Practical healthcare deployments must account for side-channel attacks (monitoring power or timing during decryption) and lattice attacks.

Important Algorithms
Paillier (PHE): Highlighted for its additive properties, making it highly efficient for basic statistical aggregation of patient records without high overhead.

CKKS & TFHE (FHE): CKKS is noted for approximate arithmetic (crucial for ML), while TFHE is noted for being bootstrap-friendly, allowing for continuous noise reduction in infinite computations.

BGV & GSW (Leveled FHE): Utilized for their modulus switching and approximate eigenvector methods, providing faster homomorphic operations for leveled circuits.

Optimizations & Implementation Strategies for MIMIC-III and MIMIC-IV
Because MIMIC datasets are massive, highly dimensional, and heavily used for deep learning (e.g., predicting ICU mortality or sepsis onset), you should apply this paper's insights in the following ways:

Utilize FLHE for Deep Neural Networks: MIMIC-based predictive models often use deep learning architectures like LSTMs or multi-layer perceptrons. Instead of using traditional FHE, implement FLHE (Fully Leveled HE). By predetermining the exact depth (number of layers) of your predictive model, FLHE can process the encrypted MIMIC features without relying on computationally disastrous bootstrapping, significantly speeding up training and inference.

Secure Federated Learning (FL): The paper emphasizes HE's role in federated learning. You can use MIMIC data to simulate a multi-hospital environment where each "hospital" (data partition) trains a local model. Use FLHE to encrypt the model gradients before aggregating them centrally. This demonstrates how a model can learn from MIMIC's massive cohort without pooling the underlying EHRs.

Defense Against Side-Channel Attacks in the ICU: If you are deploying an HE-driven model for real-time predictions (e.g., monitoring a patient's vitals on an edge device in the ICU), the paper warns of hardware-level vulnerabilities like side-channel and fault injection attacks. You must optimize not just the algorithm, but the deployment environment by adding hardware obfuscation, randomizing computation times, or using trusted execution environments (TEEs) to prevent attackers from inferring the private keys from the edge device's power consumption.

Targeted Use of PHE for Cohort Discovery: If the goal is not complex prediction but rather querying the MIMIC database for cohort statistics (e.g., "How many patients in the ICU had a specific dosage of Vasopressin?"), avoid FHE entirely. Use the Paillier algorithm (PHE) to perform additive counts on the encrypted database, drastically reducing query latency.


Research Paper 3:  Homomorphic encryption for secure and scalable predictive
healthcare analytics: a review and case study

This paper by Gogoi and Valan (2026) reviews the application of Homomorphic Encryption (HE) to protect patient confidentiality in predictive healthcare analytics, specifically in multi-institutional settings. It explores how integrating HE with decentralized technologies like federated learning and blockchain can enhance transparency and scalability. To prove its practical feasibility, the authors conducted a case study using Paillier encryption on a logistic regression model to predict heart disease.

The critical finding is that the encrypted model perfectly preserved diagnostic utility, achieving 90.16% accuracy—identical to the unencrypted plaintext model. However, this privacy came at the cost of severe computational overhead, pushing inference times from milliseconds to minutes.

Key Points for Implementing on MIMIC-III / MIMIC-IV (Minor Project)
If you are using this paper as a foundation for a minor project using large, complex Electronic Health Record (EHR) datasets like MIMIC, here is how you should structure your implementation:

Adopt Paillier for Logistic Regression: Replicate the paper's core methodology by using Paillier encryption (a Partially Homomorphic Encryption scheme). Build a binary classification model (e.g., predicting ICU mortality or 30-day readmission) using Logistic Regression. Paillier is well-suited for the additive operations required in linear and logistic regression and is much easier to implement for a minor project than Fully Homomorphic Encryption (FHE).

Aggressive Feature Selection is Mandatory: The paper explicitly warns that encrypted inference takes minutes even for a standard heart disease dataset. MIMIC has thousands of features (vitals, labs, demographics). If you encrypt all of them, your computational overhead will be insurmountable. You must perform rigorous feature selection (e.g., using Random Forest feature importance or PCA on plaintext data) to narrow down the dataset to the top 15–20 predictive clinical variables before applying encryption.

Simulate Federated Learning (FL): The authors highlight the integration of HE and federated learning to solve multi-institutional scalability. For your project, partition the MIMIC dataset by Care Unit (e.g., treat the Medical ICU, Surgical ICU, and Cardiac ICU as three separate "hospitals"). Train the logistic regression models locally on plaintext data, and use Paillier encryption only to encrypt the model weights when sending them to a central server for aggregation. This dramatically reduces latency compared to encrypting patient rows.

Benchmark the Privacy-Performance Trade-off: Align your project's results with the paper's findings. Your goal shouldn't be real-time speed, but rather proving accuracy parity. Structure your final project report to show that your encrypted MIMIC model achieves the exact same Area Under the Curve (AUC) / Accuracy as the plaintext baseline, while explicitly measuring and visualizing the added latency (CPU time and memory usage).


Research Paper 4: Homomorphic encryption for secure healthcare
artificial intelligence

Based on the active document, Homomorphic encryption for secure healthcare artificial intelligence by Yanez and Yadav, this paper presents a Systematization of Knowledge (SoK) that maps Homomorphic Encryption (HE) schemes to specific healthcare threat models, AI pipeline stages, and deployment architectures (edge vs. cloud). It concludes with a structured decision matrix to help practitioners balance security with latency and accuracy constraints.For a minor project utilizing large datasets like MIMIC-III and MIMIC-IV, the goal is to build a functional proof-of-concept without getting bottlenecked by the massive computational overhead of HE. Here are the key strategies for your implementation:Target the Inference Phase, Not Training: The paper highlights that training models entirely on encrypted data remains heavily resource-prohibitive. For a minor project, train your predictive model (e.g., predicting ICU mortality or sepsis) using plaintext MIMIC data. Apply HE exclusively during the inference phase—encrypting only the final test patient records to demonstrate a secure prediction pipeline.Simulate Edge-to-Cloud Telemetry: The authors emphasize securing data generated by distributed, resource-constrained edge devices. Structure your project into two distinct scripts: an "Edge Client" (simulating a local ICU monitor that extracts and encrypts a MIMIC patient's vitals) and a "Cloud Server" (which receives the ciphertext, runs the prediction, and returns an encrypted risk score). This directly addresses the paper's data-in-transit and data-in-use threat models.Utilize Lightweight HE Schemes (PHE/SHE): The paper classifies schemes by their suitability for specific constraints. Avoid the extreme complexity and latency of Fully Homomorphic Encryption (FHE) and bootstrapping. Instead, build a Logistic Regression model and use Partially Homomorphic Encryption (PHE, like Paillier) or Somewhat Homomorphic Encryption (SHE, like CKKS via the TenSEAL library) to handle the required arithmetic efficiently.Apply the Decision Matrix: Use the paper's proposed decision matrix as the architectural blueprint for your project. Explicitly document your chosen pathway (e.g., Tabular EHR Data $\rightarrow$ Cloud Inference $\rightarrow$ Linear Model $\rightarrow$ SHE) in your project report to justify your design choices.Focus on Benchmarking Trade-offs: The paper centers on the friction between latency, accuracy, and computational cost. Your minor project’s primary deliverable should be a benchmark evaluation proving that your encrypted MIMIC pipeline achieves the exact same diagnostic accuracy (e.g., AUC/ROC) as a plaintext baseline, while actively measuring and graphing the added CPU time and memory overhead.

