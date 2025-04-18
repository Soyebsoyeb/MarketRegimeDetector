#(1)  Market Regime Detector

A Python-based tool for detecting and analyzing different market regimes in cryptocurrency order book data using unsupervised machine learning techniques.

## Features

- Processes high-frequency order book data (up to 20 levels)
- Extracts comprehensive market microstructure features
- Implements both K-Means and Gaussian Mixture Models for clustering
- Provides regime characterization and transition analysis
- Includes visualization tools for regime exploration
- Handles multiple data files in batch processing

## Installation

Clone the repository:
git clone  https://github.com/Soyebsoyeb/MarketRegimeDetector.git
cd MarketRegimeDetector

#(2) Install required packages:
pip install -r requirements.txt

#(3) Requirements

Python 3.8+
Required packages:
numpy
pandas
scikit-learn
umap-learn
matplotlib
seaborn

#(4) For data visit the link (I couldn't upload the files after much try as the files are much bigger but I have provided the link to extract the data)

https://drive.google.com/drive/folders/1gFLwPLTE0nUN-MHoOn5u_1yrlbpI3Fst  (DATA LINK)


#(5) Command line for running the program

 python3 market_regime_detector.py


#(6) Output Analysis

<img width="1680" alt="Screenshot 2025-04-19 at 12 43 36 AM" src="https://github.com/user-attachments/assets/002e33d7-2f81-4bbd-8546-6ad094d67fbe" />





(1) UMAP Projection

UMAP Plot
2D representation of high-dimensional clusters
Points colored by regime
Ideal: Distinct groupings with minimal overlap




(2) Price Colored by Regime

Price Plot
Price series with points colored by regime
Shows which regimes occur at different price levels



(3)  Regime Evolution

Evolution Plot
Black line: Price trajectory
Colored dots: Regime occurrences
Reveals temporal patterns (e.g., volatile regimes during price swings)
