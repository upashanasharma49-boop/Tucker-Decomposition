# Tucker-Decomposition
This repository contains my work in Tucker Decomposition.
## Table of Contents
- [About] (#about)
- [Methodology] (#methodology)
- [Features] (#features)
## About
Tucker decomposition is a fundamental method in multilinear algebra for analyzing higher-order tensors. It generalizes the
singular value decomposition (SVD) from matrices to tensors by expressing a given tensor as a core tensor multiplied by a set
of orthogonal factor matrices along each mode. The core tensor captures the interactions among different components, while
the factor matrices provide the orthogonal bases in each mode. This decomposition is widely applied in signal processing, data
compression, and machine learning, as it reduces dimensionality while preserving the essential structure of the data.
## Methodology
1. **Problem Definition**  
   - The goal is to perform Tucker decomposition on a 3D tensor (l × m × n).  
   - This reduces dimensionality and extracts latent structures from data.
2. **Decomposition**  
   - Apply Tucker decomposition by factorizing the tensor into:
     - A **core tensor** (G)
     - A set of **factor matrices** (A, B, C) corresponding to each mode.
3.  **Reconstruction**  
   - The original tensor is approximated by combining the core tensor and factor matrices.  
   - Reconstruction error is computed to measure accuracy.
4. **Evaluation**  
   - Compare approximation quality with the original tensor.  
   - Analyze compression and performance.
## Features
Tucker decomposition is used to compress data, remove noise, and extract meaningful features. It reduces huge datasets into manageable representations without losing critical structural or functional information. Compared to simple decompositions, Tucker offers flexibility since the ranks can differ across modes, allowing better adaptation to real-world data.
