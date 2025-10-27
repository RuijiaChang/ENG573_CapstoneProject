'''
This script is used to predict future SpO2 and RR for multiple time windows
Author: Ruijia Chang
Date: 2025-09-28
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

class MultiWindowFuturePrediction:
    """
    Multi-window future prediction model for SpO2 and RR
    """
    
    def __init__(self, future_offsets=[10, 20, 30, 40, 50, 60]):
        """
        Initialize multi-window future prediction model
        
        Args:
            future_offsets: List of prediction time offsets in seconds
        """
        self.future_offsets = sorted(future_offsets)
        self.max_offset = max(self.future_offsets)
        
        # Store models for each time window
        self.models_spo2 = {}
        self.models_rr = {}
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def prepare_training_data(self, input_csv):
        """
        Prepare training data for multiple time windows: historical PPG features → future SpO2/RR
        
        Args:
            input_csv: Input feature file path
        
        Returns:
            dict: Training data for each time window
        """
        print(f"Preparing training data for multiple time windows: {self.future_offsets}s...")
        
        # Read data
        df = pd.read_csv(input_csv)
        print(f"Original data length: {len(df)}")
        
        # Enhanced feature engineering for better long-term prediction
        df['SpO2_trend'] = df['SpO2(mean)'].diff().fillna(0)
        df['RR_trend'] = df['RR(mean)'].diff().fillna(0)
        
        # Multiple moving averages for different time scales
        df['SpO2_ma3'] = df['SpO2(mean)'].rolling(window=3, min_periods=1).mean()
        df['SpO2_ma5'] = df['SpO2(mean)'].rolling(window=5, min_periods=1).mean()
        df['SpO2_ma10'] = df['SpO2(mean)'].rolling(window=10, min_periods=1).mean()
        
        df['RR_ma3'] = df['RR(mean)'].rolling(window=3, min_periods=1).mean()
        df['RR_ma5'] = df['RR(mean)'].rolling(window=5, min_periods=1).mean()
        df['RR_ma10'] = df['RR(mean)'].rolling(window=10, min_periods=1).mean()
        
        # Volatility features
        df['SpO2_volatility'] = df['SpO2(mean)'].rolling(window=5, min_periods=1).std().fillna(0)
        df['RR_volatility'] = df['RR(mean)'].rolling(window=5, min_periods=1).std().fillna(0)
        
        # Momentum features (rate of change)
        df['SpO2_momentum'] = df['SpO2(mean)'].diff(2).fillna(0)
        df['RR_momentum'] = df['RR(mean)'].diff(2).fillna(0)
        
        # Prepare training data for each time window
        training_data = {}
        
        for offset in self.future_offsets:
            print(f"Preparing data for {offset}s prediction...")
            
            X = []  # Input features
            y_spo2 = []  # Future SpO2 targets
            y_rr = []    # Future RR targets
            
            # Ensure sufficient data for future prediction
            max_index = len(df) - offset
            
            for i in range(max_index):
                # Create enhanced features with historical context
                current_features = df.iloc[i].to_dict()
                
            # Use enhanced features without historical lag features for simplicity
                
                # Output: future moment target values
                future_row = df.iloc[i + offset]
                future_spo2 = future_row['SpO2(mean)']
                future_rr = future_row['RR(mean)']
                
                X.append(current_features)
                y_spo2.append(future_spo2)
                y_rr.append(future_rr)
            
            training_data[offset] = {
                'X': X,
                'y_spo2': y_spo2,
                'y_rr': y_rr
            }
            
            print(f"  {offset}s samples: {len(X)}")
            print(f"  SpO2 range: {min(y_spo2):.2f} - {max(y_spo2):.2f}")
            print(f"  RR range: {min(y_rr):.2f} - {max(y_rr):.2f}")
        
        return training_data
    
    def train(self, training_data):
        """
        Train multi-window future prediction models with adaptive parameters
        
        Args:
            training_data: Dictionary containing training data for each time window
        """
        print("Starting multi-window future prediction model training...")
        
        # Use the first time window to determine feature names and scaler
        first_offset = self.future_offsets[0]
        X_first = pd.DataFrame(training_data[first_offset]['X'])
        self.feature_names = list(X_first.columns)
        
        # For tree-based models (XGBoost), scaling is not required
        self.scaler = None
        
        # Train models for each time window with adaptive parameters
        for offset in self.future_offsets:
            print(f"Training models for {offset}s prediction...")
            
            # Prepare data
            X = pd.DataFrame(training_data[offset]['X'])
            y_spo2 = training_data[offset]['y_spo2']
            y_rr = training_data[offset]['y_rr']
            
            # Adaptive XGBoost parameters based on prediction window
            if offset <= 20:
                # Short-term prediction: faster learning, less regularization
                xgb_params = dict(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=0.5,
                    objective='reg:squarederror',
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )
            elif offset <= 40:
                # Medium-term prediction: balanced parameters
                xgb_params = dict(
                    n_estimators=600,
                    max_depth=4,
                    learning_rate=0.06,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective='reg:squarederror',
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )
            else:
                # Long-term prediction: more regularization, slower learning
                xgb_params = dict(
                    n_estimators=800,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_lambda=2.0,
                    objective='reg:squarederror',
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )
            
            model_spo2 = XGBRegressor(**xgb_params)
            model_rr = XGBRegressor(**xgb_params)
            
            # Train models
            model_spo2.fit(X.values, y_spo2)
            model_rr.fit(X.values, y_rr)
            
            # Store models
            self.models_spo2[offset] = model_spo2
            self.models_rr[offset] = model_rr
            
            print(f"  {offset}s models trained successfully")
        
        self.is_trained = True
        print("Multi-window model training completed!")
    
    def predict(self, current_features):
        """
        Predict future SpO2 and RR for all time windows
        
        Args:
            current_features: Current moment features
        
        Returns:
            dict: Prediction results for all time windows
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Predict for all time windows
        predictions = {}
        
        for offset in self.future_offsets:
            # Use feature selection - only features that exist in training
            selected_features = {k: v for k, v in current_features.items() 
                               if k in self.feature_names}
            
            # Fill missing features with default values
            for feature in self.feature_names:
                if feature not in selected_features:
                    selected_features[feature] = 0.0
            
            # Convert to DataFrame
            features_df = pd.DataFrame([selected_features])
            
            # Ensure correct column order
            features_df = features_df[self.feature_names]
            
            # No scaling for XGBoost
            X_input = features_df.values
            
            # Prediction for all models
            spo2_pred = float(self.models_spo2[offset].predict(X_input)[0])
            rr_pred = float(self.models_rr[offset].predict(X_input)[0])
            spo2_std = None
            rr_std = None
            
            # Apply physiological constraints
            spo2_pred = np.clip(spo2_pred, 70, 100)
            rr_pred = np.clip(rr_pred, 8, 40)
            
            predictions[offset] = {
                'future_spo2': spo2_pred,
                'future_rr': rr_pred,
                'confidence_spo2': spo2_std,
                'confidence_rr': rr_std
            }
        
        return predictions
    
    def evaluate(self, test_data):
        """
        Evaluate model performance for all time windows
        
        Args:
            test_data: Dictionary containing test data for each time window
        
        Returns:
            dict: Evaluation metrics for all time windows
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        evaluation_results = {}
        
        for offset in self.future_offsets:
            print(f"Evaluating {offset}s prediction...")
            
            # Prepare test data
            X_test = pd.DataFrame(test_data[offset]['X'])
            y_spo2_test = test_data[offset]['y_spo2']
            y_rr_test = test_data[offset]['y_rr']
            
            # Predict (no scaling required)
            spo2_pred = self.models_spo2[offset].predict(X_test.values)
            rr_pred = self.models_rr[offset].predict(X_test.values)
            
            # Calculate evaluation metrics
            spo2_mae = mean_absolute_error(y_spo2_test, spo2_pred)
            spo2_rmse = np.sqrt(mean_squared_error(y_spo2_test, spo2_pred))
            spo2_r2 = r2_score(y_spo2_test, spo2_pred)
            
            rr_mae = mean_absolute_error(y_rr_test, rr_pred)
            rr_rmse = np.sqrt(mean_squared_error(y_rr_test, rr_pred))
            rr_r2 = r2_score(y_rr_test, rr_pred)
            
            evaluation_results[offset] = {
                'SpO2_MAE': spo2_mae,
                'SpO2_RMSE': spo2_rmse,
                'SpO2_R2': spo2_r2,
                'RR_MAE': rr_mae,
                'RR_RMSE': rr_rmse,
                'RR_R2': rr_r2,
                'predictions': {
                    'spo2_pred': spo2_pred,
                    'rr_pred': rr_pred,
                    'spo2_true': y_spo2_test,
                    'rr_true': y_rr_test
                }
            }
        
        return evaluation_results
    
    def save_model(self, filepath):
        """
        Save trained multi-window model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        model_data = {
            'future_offsets': self.future_offsets,
            'max_offset': self.max_offset,
            'models_spo2': self.models_spo2,
            'models_rr': self.models_rr,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Multi-window model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained multi-window model
        """
        model_data = joblib.load(filepath)
        
        self.future_offsets = model_data['future_offsets']
        self.max_offset = model_data['max_offset']
        self.models_spo2 = model_data['models_spo2']
        self.models_rr = model_data['models_rr']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Multi-window model loaded from {filepath}")

def main():
    """
    Main function: demonstrate multi-window future prediction
    """
    print("=== Multi-Window Future Prediction Demo ===")
    
    # Set parameters
    future_offsets = [10, 20, 30, 40, 50, 60]  # Multiple prediction time windows
    input_csv = "BIDMC_Regression/features/BIDMC_Segmented_features.csv"
    
    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"Error: File not found {input_csv}")
        print("Please ensure BIDMC_Segmented_features.csv exists")
        return
    
    # Initialize model
    model = MultiWindowFuturePrediction(future_offsets)
    
    # Prepare training data
    print("Preparing training data...")
    training_data = model.prepare_training_data(input_csv)
    
    # Split training and test data for each time window
    test_data = {}
    for offset in future_offsets:
        X = training_data[offset]['X']
        y_spo2 = training_data[offset]['y_spo2']
        y_rr = training_data[offset]['y_rr']
        
        X_train, X_test, y_spo2_train, y_spo2_test, y_rr_train, y_rr_test = train_test_split(
            X, y_spo2, y_rr, test_size=0.2, random_state=42
        )
        
        # Update training data with train split
        training_data[offset] = {
            'X': X_train,
            'y_spo2': y_spo2_train,
            'y_rr': y_rr_train
        }
        
        # Store test data
        test_data[offset] = {
            'X': X_test,
            'y_spo2': y_spo2_test,
            'y_rr': y_rr_test
        }
        
        print(f"{offset}s - Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    model.train(training_data)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    evaluation_results = model.evaluate(test_data)
    
    # Display results for each time window
    print(f"\n=== Multi-Window Prediction Performance Evaluation ===")
    print(f"{'Time Window':<12} {'SpO2 MAE':<10} {'SpO2 R²':<10} {'RR MAE':<10} {'RR R²':<10}")
    print("-" * 60)
    
    # Store results for analysis
    spo2_maes = []
    rr_maes = []
    
    for offset in future_offsets:
        eval_result = evaluation_results[offset]
        spo2_maes.append(eval_result['SpO2_MAE'])
        rr_maes.append(eval_result['RR_MAE'])
        print(f"{offset}s{'':<8} {eval_result['SpO2_MAE']:<10.4f} {eval_result['SpO2_R2']:<10.4f} "
              f"{eval_result['RR_MAE']:<10.4f} {eval_result['RR_R2']:<10.4f}")
    
    # Performance analysis
    print(f"\n=== Performance Analysis ===")
    spo2_maes = np.array(spo2_maes)
    rr_maes = np.array(rr_maes)
    
    print(f"SpO2 MAE - Best: {min(spo2_maes):.4f} ({future_offsets[np.argmin(spo2_maes)]}s), "
          f"Worst: {max(spo2_maes):.4f} ({future_offsets[np.argmax(spo2_maes)]}s)")
    print(f"RR MAE - Best: {min(rr_maes):.4f} ({future_offsets[np.argmin(rr_maes)]}s), "
          f"Worst: {max(rr_maes):.4f} ({future_offsets[np.argmax(rr_maes)]}s)")
    
    # Identify problematic windows
    spo2_threshold = np.mean(spo2_maes) + np.std(spo2_maes)
    rr_threshold = np.mean(rr_maes) + np.std(rr_maes)
    
    problematic_windows = []
    for i, offset in enumerate(future_offsets):
        if spo2_maes[i] > spo2_threshold or rr_maes[i] > rr_threshold:
            problematic_windows.append(offset)
    
    if problematic_windows:
        print(f"Problematic windows (MAE > mean+std): {problematic_windows}")
        print("These windows may need special attention or different modeling approaches.")
    
    # Show prediction examples for different time windows
    print(f"\n=== Multi-Window Prediction Examples ===")
    
    # Use first time window's test data for examples (all time windows have same input features)
    first_offset = future_offsets[0]
    X_test_examples = test_data[first_offset]['X']
    
    for i in range(min(3, len(X_test_examples))):
        print(f"\nSample {i+1}:")
        
        # For demonstration, create some mock historical data
        # In real usage, this would come from previous measurements
        historical_data = []
        if i > 0:
            # Use previous samples as historical context
            for j in range(max(1, i-5), i):
                if j < len(X_test_examples):
                    historical_data.append(X_test_examples[j])
        
        predictions = model.predict(X_test_examples[i])
        
        for offset in future_offsets:
            pred = predictions[offset]
            # Get true values for this specific time window
            true_spo2 = test_data[offset]['y_spo2'][i]
            true_rr = test_data[offset]['y_rr'][i]
            
            print(f"  {offset}s prediction:")
            
            # SpO2 prediction with confidence level (N/A for XGBoost)
            if pred['confidence_spo2'] is None:
                print(f"    SpO2: {pred['future_spo2']:.2f}% (confidence: N/A)")
            else:
                spo2_conf_level = "High" if pred['confidence_spo2'] < 0.5 else "Medium" if pred['confidence_spo2'] < 1.0 else "Low"
                print(f"    SpO2: {pred['future_spo2']:.2f}% (confidence: {pred['confidence_spo2']:.3f} - {spo2_conf_level})")
            print(f"    True SpO2: {true_spo2:.2f}% (error: {abs(true_spo2 - pred['future_spo2']):.2f}%)")
            
            # RR prediction with confidence level (N/A for XGBoost)
            if pred['confidence_rr'] is None:
                print(f"    RR: {pred['future_rr']:.2f} breaths/min (confidence: N/A)")
            else:
                rr_conf_level = "High" if pred['confidence_rr'] < 0.5 else "Medium" if pred['confidence_rr'] < 1.0 else "Low"
                print(f"    RR: {pred['future_rr']:.2f} breaths/min (confidence: {pred['confidence_rr']:.3f} - {rr_conf_level})")
            print(f"    True RR: {true_rr:.2f} breaths/min (error: {abs(true_rr - pred['future_rr']):.2f})")
            print()
    
    # Save model
    model_path = f"multi_window_prediction_model_{'-'.join(map(str, future_offsets))}s.pkl"
    model.save_model(model_path)
    
    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
