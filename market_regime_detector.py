import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.feature_names = []
        
    def load_data(self, file_path, data_type='depth'):
        """Load and parse the order book data with proper timestamp handling"""
        try:
            # Read the data with proper handling of the timestamp format
            df = pd.read_csv(file_path)
            
            # Clean and parse the timestamp column
            df['timestamp'] = df['Time'].str.extract(r'^([^+]+)')[0].str.strip()
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            
            # Drop rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Drop the original Time column
            df = df.drop(columns=['Time'])
            
            # Convert all numeric columns to float
            numeric_cols = [col for col in df.columns if 'Price' in col or 'Qty' in col]
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

    def process_depth_data(self, depth_df):
        """Process the wide format depth data"""
        return depth_df

    def merge_data(self, depth_data, trade_data=None):
        """Process depth data (trade data is optional)"""
        if isinstance(depth_data, str):
            depth_df = self.load_data(depth_data, 'depth')
            depth_df = self.process_depth_data(depth_df)
        else:
            depth_df = depth_data.copy()
            
        if trade_data is not None:
            if isinstance(trade_data, str):
                trade_df = self.load_data(trade_data, 'trade')
            else:
                trade_df = trade_data.copy()
                
            if depth_df.index.name != 'timestamp':
                depth_df = depth_df.reset_index()
            if 'timestamp' not in trade_df.columns:
                raise ValueError("Trade data must contain timestamp information")
                
            depth_df.set_index('timestamp', inplace=True)
            trade_df.set_index('timestamp', inplace=True)
            
            trade_resampled = trade_df.resample('1S').agg({
                'price': 'mean',
                'qty': 'sum',
                'isBuyerMaker': 'first'
            })
            trade_resampled['buy_volume'] = np.where(trade_resampled['isBuyerMaker'] == False, 
                                                   trade_resampled['qty'], 0)
            trade_resampled['sell_volume'] = np.where(trade_resampled['isBuyerMaker'] == True, 
                                                    trade_resampled['qty'], 0)
            
            merged = pd.merge_asof(
                depth_df.sort_index(),
                trade_resampled.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest'
            )
            
            trade_cols = ['price', 'qty', 'buy_volume', 'sell_volume']
            merged[trade_cols] = merged[trade_cols].ffill()
            
            return merged.dropna()
        
        return depth_df

    def engineer_features(self, df):
        """Create comprehensive market regime features from order book data"""
        features = pd.DataFrame(index=df.index)
        
        # Extract order book levels (1-20)
        bid_prices = [df[f'BidPriceL{i}'] for i in range(1, 21)]
        bid_qtys = [df[f'BidQtyL{i}'] for i in range(1, 21)]
        ask_prices = [df[f'AskPriceL{i}'] for i in range(1, 21)]
        ask_qtys = [df[f'AskQtyL{i}'] for i in range(1, 21)]
        
        # Basic price features
        features['spread'] = ask_prices[0] - bid_prices[0]
        features['mid_price'] = (bid_prices[0] + ask_prices[0]) / 2
        features['microprice'] = (bid_prices[0]*ask_qtys[0] + ask_prices[0]*bid_qtys[0]) / (bid_qtys[0] + ask_qtys[0] + 1e-6)
        
        # Order book imbalance features
        for level in range(5):
            total_volume = bid_qtys[level] + ask_qtys[level] + 1e-6
            features[f'imbalance_lvl{level+1}'] = (bid_qtys[level] - ask_qtys[level]) / total_volume
            features[f'relative_imbalance_lvl{level+1}'] = (bid_qtys[level] - ask_qtys[level]) / total_volume
            
        # Cumulative depth features
        features['cum_bid_qty_10'] = pd.DataFrame(bid_qtys[:10]).T.sum(axis=1)
        features['cum_ask_qty_10'] = pd.DataFrame(ask_qtys[:10]).T.sum(axis=1)
        features['cum_bid_qty_20'] = pd.DataFrame(bid_qtys).T.sum(axis=1)
        features['cum_ask_qty_20'] = pd.DataFrame(ask_qtys).T.sum(axis=1)
        features['total_depth'] = features['cum_bid_qty_20'] + features['cum_ask_qty_20']
        features['depth_imbalance'] = (features['cum_bid_qty_20'] - features['cum_ask_qty_20']) / features['total_depth']
        
        # Price movement features
        features['returns'] = np.log(features['mid_price'] / features['mid_price'].shift(1))
        features['volatility_10s'] = features['returns'].rolling('10s').std()
        features['volatility_30s'] = features['returns'].rolling('30s').std()
        features['price_change_5s'] = features['mid_price'].pct_change(5)
        
        # Order book slope features
        def calc_slope(prices, qtys, n_levels=5):
            slopes = []
            for i in range(len(prices[0])):
                valid_prices = [p.iloc[i] for p in prices[:n_levels] if not np.isnan(p.iloc[i])]
                valid_qtys = [q.iloc[i] for q in qtys[:n_levels] if not np.isnan(q.iloc[i])]
                
                if len(valid_prices) >= 2 and len(valid_qtys) >= 2:
                    x = np.array(valid_prices) - valid_prices[0]
                    y = np.log(np.array(valid_qtys) + 1e-6)
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            return pd.Series(slopes, index=prices[0].index)
            
        features['bid_slope_5'] = calc_slope(bid_prices, bid_qtys, 5)
        features['ask_slope_5'] = calc_slope(ask_prices, ask_qtys, 5)
        features['bid_slope_10'] = calc_slope(bid_prices, bid_qtys, 10)
        features['ask_slope_10'] = calc_slope(ask_prices, ask_qtys, 10)
        
        # If trade data is available
        if 'price' in df.columns:
            features['trade_price'] = df['price']
            features['trade_volume'] = df['qty']
            features['buy_volume'] = df.get('buy_volume', 0)
            features['sell_volume'] = df.get('sell_volume', 0)
            features['volume_imb'] = (features['buy_volume'] - features['sell_volume']) / \
                                   (features['buy_volume'] + features['sell_volume'] + 1e-6)
            features['vol_10s'] = df['qty'].rolling('10s').sum()
            features['vol_30s'] = df['qty'].rolling('30s').sum()
            
            # VWAP features
            vwap = (df['price'] * df['qty']).rolling('30s').sum() / df['qty'].rolling('30s').sum()
            features['vwap'] = vwap
            features['vwap_spread'] = features['mid_price'] - vwap
            features['vwap_change'] = vwap.diff()
        
        # Drop any remaining NA values
        features = features.dropna()
        self.feature_names = features.columns.tolist()
        
        return features

    def preprocess(self, features):
        """Normalize and reduce dimensionality"""
        features = features.ffill().fillna(0)
        X = self.scaler.fit_transform(features)
        self.pca = PCA(n_components=0.95)
        X = self.pca.fit_transform(X)
        print(f"Reduced dimensionality from {features.shape[1]} to {X.shape[1]}")
        return X

    def cluster(self, X, method='gmm', n_clusters=4):
        """Apply clustering with automatic selection of optimal clusters"""
        print(f"\nClustering with {method} and {n_clusters} clusters...")
        
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        else:
            raise ValueError("Unsupported clustering method")
        
        self.model = model
        labels = model.fit_predict(X)
        
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            print(f"Clustering metrics - Silhouette: {silhouette:.3f}, Davies-Bouldin: {db_score:.3f}")
        else:
            print("Warning: Only one cluster found - cannot compute metrics")
        
        return labels, model

    def analyze_regimes(self, features, labels):
        """Analyze and characterize each market regime"""
        features['regime'] = labels
        stats = features.groupby('regime').agg(['mean', 'std'])
        
        regime_names = {}
        for regime in stats.index:
            vol = stats.loc[regime, ('volatility_10s', 'mean')]
            liq = stats.loc[regime, ('total_depth', 'mean')]
            spread = stats.loc[regime, ('spread', 'mean')]
            returns = stats.loc[regime, ('returns', 'mean')]
            
            vol_type = 'HighVol' if vol > features['volatility_10s'].quantile(0.75) else \
                     'LowVol' if vol < features['volatility_10s'].quantile(0.25) else 'MedVol'
            
            liq_type = 'HighLiq' if liq > features['total_depth'].quantile(0.75) else \
                      'LowLiq' if liq < features['total_depth'].quantile(0.25) else 'MedLiq'
            
            trend_type = 'Bull' if returns > 0.0005 else \
                        'Bear' if returns < -0.0005 else 'Neutral'
            
            spread_type = 'Wide' if spread > features['spread'].quantile(0.75) else \
                        'Tight' if spread < features['spread'].quantile(0.25) else 'Normal'
            
            regime_names[regime] = f"{trend_type}_{vol_type}_{liq_type}_{spread_type}"
        
        summary = pd.DataFrame({
            'regime': stats.index,
            'name': [regime_names[r] for r in stats.index],
            'volatility': stats[('volatility_10s', 'mean')],
            'liquidity': stats[('total_depth', 'mean')],
            'spread': stats[('spread', 'mean')],
            'returns': stats[('returns', 'mean')],
            'count': features['regime'].value_counts().sort_index(),
            'duration_avg': features.groupby('regime').size() / len(features)
        })
        
        return summary.sort_values('count', ascending=False)

    def visualize(self, X, labels, price_series=None, n_samples=1000):
        """Visualize clusters with optional price series"""
        plt.figure(figsize=(18, 12))
        
        if len(X) > n_samples:
            idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[idx]
            labels_sample = labels[idx]
            if price_series is not None:
                price_sample = price_series.iloc[idx]
        else:
            X_sample = X
            labels_sample = labels
            price_sample = price_series
        
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_sample)
        
        plt.subplot(2, 2, 1)
        sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels_sample, 
                        palette='viridis', alpha=0.6, s=10)
        plt.title("UMAP Projection of Market Regimes")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        
        if price_series is not None:
            plt.subplot(2, 2, 2)
            for regime in np.unique(labels_sample):
                mask = labels_sample == regime
                plt.scatter(price_sample.index[mask], price_sample[mask], 
                           label=f"Regime {regime}", alpha=0.6, s=10)
            plt.title("Price Colored by Regime")
            plt.ylabel("Price")
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(price_sample.index, price_sample, 'k-', alpha=0.3, linewidth=0.5)
            for regime in np.unique(labels_sample):
                mask = labels_sample == regime
                plt.scatter(price_sample.index[mask], price_sample[mask], 
                           label=f"Regime {regime}", alpha=0.6, s=10)
            plt.title("Regime Evolution Over Time")
            plt.ylabel("Price")
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def regime_transition_matrix(self, labels):
        """Create regime transition probability matrix"""
        unique_regimes = np.unique(labels)
        n_regimes = len(unique_regimes)
        trans_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(1, len(labels)):
            prev = np.where(unique_regimes == labels[i-1])[0][0]
            curr = np.where(unique_regimes == labels[i])[0][0]
            trans_matrix[prev, curr] += 1
        
        row_sums = trans_matrix.sum(axis=1)
        trans_matrix = trans_matrix / (row_sums[:, np.newaxis] + 1e-6)
        
        trans_df = pd.DataFrame(trans_matrix, 
                               index=[f"Regime {i}" for i in unique_regimes],
                               columns=[f"Regime {i}" for i in unique_regimes])
        
        return trans_df

def main():
    detector = MarketRegimeDetector()
    
    try:
        results_list = []
        file_pairs = [
            ("data/depth20_1000ms/BNBFDUSD_20250314.txt", None),
            ("data/depth20_1000ms/BNBFDUSD_20250315.txt", None),
            ("data/depth20_1000ms/BNBFDUSD_20250316.txt", None),
            ("data/depth20_1000ms/BNBFDUSD_20250317.txt", None)
        ]
        
        for depth_file, trade_file in file_pairs:
            print(f"\nProcessing {depth_file}...")
            
            # Load and process data
            merged = detector.merge_data(depth_file, trade_file)
            features = detector.engineer_features(merged)
            X = detector.preprocess(features)
            
            # Cluster and analyze
            labels_gmm, _ = detector.cluster(X, method='gmm', n_clusters=4)
            regime_stats = detector.analyze_regimes(features, labels_gmm)
            
            print("\nRegime Statistics:")
            print(regime_stats)
            
            # Prepare results - Filter merged data to match features index
            results = merged.loc[features.index].copy()
            results['regime'] = labels_gmm
            results['regime_name'] = results['regime'].map(regime_stats.set_index('regime')['name'])
            results['source'] = os.path.basename(depth_file)
            results_list.append(results)
            
            # Show transition matrix
            print("\nRegime Transition Matrix:")
            print(detector.regime_transition_matrix(labels_gmm))
            
            # Show visualization
            mid_price = (merged['BidPriceL1'] + merged['AskPriceL1'])/2
            mid_price = mid_price.loc[features.index]  # Ensure alignment
            detector.visualize(X, labels_gmm, mid_price)
        
        # Save all results
        final_results = pd.concat(results_list)
        final_results.to_csv('market_regimes_results.csv')
        print("\nAll processing complete. Results saved to market_regimes_results.csv")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please check your input files and ensure they have the required columns.")

if __name__ == '__main__':
    main()