# %%
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
def loadData (flow, meta):
    df1 = pd.read_csv(f'{flow}', delim_whitespace=True, header=None)
    df1.columns = ['ID', 'Year', 'Mnth', 'Day', 'Flow0', 'SYM']
    df1['datetime'] = pd.to_datetime(df1[['Year', 'Mnth', 'Day']].rename(columns={'Year': 'year', 'Mnth': 'month', 'Day': 'day'}))
    df1.drop(['ID', 'Year', 'Mnth', 'Day'], axis=1, inplace=True)
    df2 = pd.read_csv(f'{meta}', skiprows=3, delim_whitespace=True)
    df = pd.concat([df1, df2], axis=1)
    df = pd.get_dummies(df, columns=['SYM'], dummy_na=True)
    df.set_index('datetime', inplace=True)
    df.drop(['Hr'], axis=1, inplace=True)
    before = 3
    after = 7
    # for i in range(1, before + 1):
    #     df[f'Flow-{i}'] = df['Flow0'].shift(+i)  # Shift down by 1   
    for i in  range(1, after + 1):
        df[f'Flow+{i}'] = df['Flow0'].shift(-i)  # Shift up by 1   
    df = df.dropna()
    return df

# %%
def expand_mean(df):
    return df.expanding(1).mean()
def pct_diff(old, new):
    return (new-old) / old
def compute_rolling(df, horizon, col):
    label = f"rolling_{horizon}_{col}"

    df[label] = df[col].rolling(horizon).mean()
    df[f"{label}_pct"] = pct_diff(df[label], df[col])
    return df
# %%
def CAMELSrun(data_id, horizon1, horizon2, target_var, SYM_M):
    df = loadData(f'data/CAMELS/{data_id}_streamflow_qc.txt', f'data/CAMELS/{data_id}_lump_maurer_forcing_leap.txt')
    rolling_horizons = [horizon1, horizon2]

    for horizon in rolling_horizons:
        for col in ['prcp(mm/day)', 'srad(W/m2)', 'Flow0', 'tmax(C)', 'tmin(C)']:
            df = compute_rolling(df, horizon, col)

    df = df.fillna(0)

    for col in ['prcp(mm/day)', 'srad(W/m2)', 'Flow0', 'tmax(C)', 'tmin(C)']:
        df[f'month_avg_{col}'] = df[col].groupby(df.index.month, group_keys=False).apply(expand_mean)
        df[f'day_avg_{col}'] = df[col].groupby(df.index.day_of_year, group_keys=False).apply(expand_mean)

    # List of features to drop based on importance scores
    features_to_drop = [
        'SYM_A', 'SYM_A:e', 'SYM_nan', 'swe(mm)', 'vp(Pa)',
        'Year', 'Day', 'dayl(s)',
        # 'tmax(C)', 'tmin(C)', 
        # 'rolling_3_prcp(mm/day)', 'rolling_3_srad(W/m2)_pct',
        # 'rolling_3_tmax(C)_pct', 'rolling_3_tmin(C)_pct',
        # 'rolling_3_tmax(C)',
        # 'rolling_7_prcp(mm/day)', 'rolling_7_prcp(mm/day)_pct', 
        # 'rolling_7_tmax(C)', 'rolling_7_vp(Pa)',
        # 'rolling_7_tmin(C)_pct', 'rolling_7_tmax(C)_pct',
        'rolling_7_tmax(C)_pct', 'rolling_7_tmin(C)_pct',
        # 'rolling_14_srad(W/m2)_pct',
        # 'rolling_14_tmax(C)', 'rolling_14_tmin(C)', 
        # 'rolling_14_prcp(mm/day)', 'rolling_14_srad(W/m2)', 
        # 'rolling_14_tmax(C)_pct', 'rolling_14_tmin(C)_pct',
        # 'month_avg_vp(Pa)',
        # 'month_avg_tmax(C)',
        # 'month_avg_prcp(mm/day)',
        # 'month_avg_srad(W/m2)'
        # 'day_avg_vp(Pa)',
    ]

    if SYM_M:
        features_to_drop.append('SYM_M')
    # Drop the features from the DataFrame
    df = df.drop(columns=features_to_drop)

    Y = df[f'{target_var}']  # Target variable
    adjX = df.drop(['Flow+1', 'Flow+2', 'Flow+3', 'Flow+4', 'Flow+5', 'Flow+6', 'Flow+7'], axis=1)  # Features
    X = adjX.columns

    return df, X, Y
# %%
def backtest(df, model, X, Y, epochsno, train_window_size, test_window_size, drop_before_index, device, patience=15, batch_size=32):
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    all_predictions = []
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    if drop_before_index is not None:
        df = df[df.index >= drop_before_index]
    
    num_iterations = 0
    
    if isinstance(Y, pd.Series):
        Y_column = Y.name
    else:
        Y_column = Y
    
    test_indices = []
    
    for i in range(0, df.shape[0] - train_window_size - test_window_size + 1, test_window_size):
        num_iterations += 1
        
        train = df.iloc[i:(i+train_window_size), :]
        test = df.iloc[(i+train_window_size):(i+train_window_size+test_window_size), :]
        
        test_indices.extend(test.index)
        
        X_train = scaler_X.fit_transform(train[X])
        X_test = scaler_X.transform(test[X])
        Y_train = scaler_Y.fit_transform(train[[Y_column]]).reshape(-1, 1)
        Y_test = scaler_Y.transform(test[[Y_column]]).reshape(-1, 1)
        
        if device == "cpu":
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, X_train.shape[1])
            Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, X_test.shape[1])
        else: 
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, X_train.shape[1]).to(device)
            Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, X_test.shape[1]).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochsno):
            epoch_loss = 0.0
            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)  # Extract the output tensor
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            print(f"Epoch {epoch+1}/{epochsno}, Loss: {epoch_loss}")
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor)  # Extract the output tensor
            if device == "cpu":
                preds = preds.numpy()
            else:
                preds = preds.cpu().numpy()
        
        preds = scaler_Y.inverse_transform(preds).flatten()
        Y_test = scaler_Y.inverse_transform(Y_test).flatten()
        
        combined = pd.DataFrame({
            "actual": Y_test,
            "prediction": preds
        }, index=test.index)
        
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        all_predictions.append(combined)
    
    all_predictions = pd.concat(all_predictions, axis=0)
    datetime = pd.Index(test_indices)
    return all_predictions, datetime

# %%
def standardtest(df, model, X, Y, device):
    all_predictions = []
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    if isinstance(Y, pd.Series):
        Y_column = Y.name
    else:
        Y_column = Y

    # Scale features
    X_test = df[X]
    X_test = scaler_X.fit_transform(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)  # Convert to PyTorch tensor

    # Scale target variable
    Y_test = df[[Y_column]]
    Y_test = scaler_Y.fit_transform(Y_test).reshape(-1, 1)
    
    # Convert to tensor
    # Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

    # Make predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()

    # Inverse transform predictions and actual values
    preds = scaler_Y.inverse_transform(preds).reshape(-1, 1).flatten()
    # Y_test = scaler_Y.inverse_transform(Y_test_tensor.numpy()).reshape(-1, 1).flatten()

    # Store results in DataFrame
    combined = pd.DataFrame({
        "actual": Y_test,
        "prediction": preds
    })

    combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
    all_predictions.append(combined)

    all_predictions = pd.concat(all_predictions, axis=0)

    # Create a datetime index for all test predictions
    datetime = df.index

    return all_predictions, datetime

def visualization(datetime, y_pred, y_test, zoom_start, zoom_end):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = 100 - np.mean(mape)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {round(mae, 2)} m3/s")
    print(f"MSE: {round(mse, 2)} m3/s")
    print(f"RMSE: {round(np.sqrt(mse), 2)} m3/s")
    print(f"MAPE: {round(mape, 2)}%")
    print(f"Accuracy: {round(accuracy, 3)}%")
    print(f"R2: {round(r2, 4)}")
    
    # Plot Actual vs Predicted Streamflow Values
    plt.figure(figsize=(10, 6))
    plt.plot(datetime, y_test, label='Actual Flow', color='black')
    plt.plot(datetime, y_pred, label='Predicted Flow', linestyle='--', color='C0')
    plt.xlabel('Datetime')
    plt.ylabel('Streamflow (m3/s)')
    plt.title('Actual vs Predicted Streamflow Values')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Zoomed in Plot
    plt.figure(figsize=(10, 6))
    plt.plot(datetime, y_test, label='Actual Flow', color='black')
    plt.plot(datetime, y_pred, label='Predicted Flow', color='C0')
    plt.xlabel('Datetime')
    plt.ylabel('Streamflow (m3/s)')
    plt.title('Actual vs Predicted Streamflow Values (Zoomed)')
    
    # Set the x and y limits to zoom in
    plt.xlim(datetime[zoom_start], datetime[zoom_end])
    plt.ylim(min(y_test[zoom_start:zoom_end].min(), y_pred[zoom_start:zoom_end].min()), 
             max(y_test[zoom_start:zoom_end].max(), y_pred[zoom_start:zoom_end].max()))
    
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Residuals Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.plot(datetime, residuals, label='Residuals Flow+0')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Datetime')
    plt.ylabel('Residual (m3/s)')
    plt.title('Residuals for Each Day Ahead Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Residuals Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, label='Residuals for Flow+0')
    plt.axhline(0, color='black', linestyle='--', label='Zero Residual Line')
    plt.xlabel('Predicted Streamflow (m3/s)')
    plt.ylabel('Residual (m3/s)')
    plt.title('Residual Scatter Plot for Each Day Ahead Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Actual vs Predicted Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, label='Flow+0', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--', label='Ideal Fit')
    plt.xlabel('Actual Streamflow (m3/s)')
    plt.ylabel('Predicted Streamflow (m3/s)')
    plt.title('Actual vs Predicted Streamflow Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
def permutation_importance(model, X_test, y_test, metric=mean_squared_error):
    baseline_score = metric(y_test, model.predict(X_test))
    importances = []
    
    for i in range(X_test.shape[1]):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        permuted_score = metric(y_test, model.predict(X_test_permuted))
        importances.append(permuted_score - baseline_score)
    
    return np.array(importances)

def print_model_weights_and_biases(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}\n{param.data}")
