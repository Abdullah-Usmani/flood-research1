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
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.layers import Dense
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

# rolling_horizons = [3, 7]

# for horizon in rolling_horizons:
#     for col in ['prcp(mm/day)', 'srad(W/m2)', 'Flow0', 'tmax(C)', 'tmin(C)', 'vp(Pa)']:
#         df = compute_rolling(df, horizon, col)

# df = df.fillna(0)

# for col in ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'Flow0', 'tmin(C)', 'vp(Pa)']:
#     df[f'month_avg_{col}'] = df[col].groupby(df.index.month, group_keys=False).apply(expand_mean)
#     df[f'day_avg_{col}'] = df[col].groupby(df.index.day_of_year, group_keys=False).apply(expand_mean)

# %%
def CAMELSrun(data_id, horizon1, horizon2, target_var, SYM_M):
    df = loadData(f'Data/Idaho/{data_id}_streamflow_qc.txt', f'Data/Idaho/{data_id}_lump_maurer_forcing_leap.txt')
    rolling_horizons = [horizon1, horizon2]

    for horizon in rolling_horizons:
        for col in ['prcp(mm/day)', 'srad(W/m2)', 'Flow0', 'tmax(C)', 'tmin(C)', 'vp(Pa)']:
            df = compute_rolling(df, horizon, col)

    df = df.fillna(0)

    for col in ['prcp(mm/day)', 'srad(W/m2)', 'Flow0', 'tmax(C)', 'tmin(C)', 'vp(Pa)']:
        df[f'month_avg_{col}'] = df[col].groupby(df.index.month, group_keys=False).apply(expand_mean)
        df[f'day_avg_{col}'] = df[col].groupby(df.index.day_of_year, group_keys=False).apply(expand_mean)

    # List of features to drop based on importance scores
    features_to_drop = [
        'SYM_A', 'SYM_A:e', 'SYM_nan', 'swe(mm)',
        'Year', 'Day', 
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


# # %%
# df, X, Y = CAMELSrun('13340000', 3, 7, 'Flow+2', False)

# # %%
# model = keras.Sequential([
#     Dense(32, activation='relu', input_shape=(X.shape[0],), kernel_regularizer='l2'),
#     Dropout(0.1),
#     Dense(16, activation='relu', kernel_regularizer='l2'),
#     Dropout(0.1),
#     Dense(1)
# ])


# model.compile(optimizer='adam',
#               loss='mean_squared_error',
#               metrics=['mean_squared_error'])


# # Summary to see the architecture
# model.summary()


# %%
def print_model_weights_and_biases(model):
    for layer_idx, layer in enumerate(model.layers):
        weights = layer.get_weights()  # Returns a list: [weights, biases]
        print(f"Layer {layer_idx+1}: {layer.name}")
        
        if len(weights) > 0:  # Some layers (like Dropout) may not have weights
            print(f"  Weights shape: {weights[0].shape}")
            print(f"  Weights: {weights[0]}")
            
            if len(weights) > 1:  # If biases exist
                print(f"  Biases shape: {weights[1].shape}")
                print(f"  Biases: {weights[1]}")
        else:
            print("  No weights/biases for this layer.")

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

# %%
def backtest(df, model, X, Y, train_window_size, test_window_size, drop_before_index):
    all_predictions = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    # feature_importances = np.zeros(len(X))
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    # Optionally drop rows before the specified index
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

        X_train = train[X]
        X_train = scaler_X.fit_transform(X_train)

        X_test = test[X]
        X_test = scaler_X.transform(X_test)
        
        Y_train = train[[Y_column]]
        Y_train = scaler_Y.fit_transform(Y_train).reshape(-1, 1) 

        Y_test = test[[Y_column]]
        Y_test = scaler_Y.transform(Y_test).reshape(-1, 1) 

        model.fit(X_train, Y_train, epochs=15, verbose=1, validation_split=0.2, callbacks=[early_stopping])

        preds = model.predict(X_test)

        preds = scaler_Y.inverse_transform(preds).reshape(-1, 1).flatten() 
        Y_test = scaler_Y.inverse_transform(Y_test).reshape(-1, 1).flatten()

        combined = pd.DataFrame({
            "actual": Y_test,
            "prediction": preds
        }, index=test.index)

        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        all_predictions.append(combined)

    #     importances = permutation_importance(model, X_test, Y_test)
    #     feature_importances += importances


    # avg_feature_importances = feature_importances / num_iterations
    
    # for i, importance in enumerate(avg_feature_importances):
    #     print(f"Feature: {X[i]}, Importance: {importance}")

    all_predictions = pd.concat(all_predictions, axis=0)

    # Create a datetime index for all test predictions
    datetime = pd.Index(test_indices)

    return all_predictions, datetime

# %%
def standardtest(df, model, X, Y):
    all_predictions = []
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    # Y = np.log1p(Y)

    if isinstance(Y, pd.Series):
        Y_column = Y.name
    else:
        Y_column = Y

    X_test = df[X]
    X_test = scaler_X.fit_transform(X_test)

    # Y_test = np.log1p(test[[Y_column]])
    Y_test = df[[Y_column]]
    Y_test = scaler_Y.fit_transform(Y_test).reshape(-1, 1) 

    preds = model.predict(X_test)

    preds = scaler_Y.inverse_transform(preds).reshape(-1, 1).flatten() 
    Y_test = scaler_Y.inverse_transform(Y_test).reshape(-1, 1).flatten()

    # preds = np.expm1(preds)  # Reverse the log1p transformation.
    # Y_test = np.expm1(Y_test)  # Reverse the log1p transformation for the test data as well.

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


# # %%
# df, X, Y = CAMELSrun('04045500', 3, 7, 'Flow+7', True)

# # %%
# predictions, datetime1 = backtest(df, model, X, Y, 540, 60, None)
# y_pred = predictions["prediction"]
# y_test = predictions["actual"]

# y_pred = np.maximum(y_pred, 0)  # Ensure no negative predictions
# y_pred = np.minimum(y_pred, np.max(y_test) * 1.1)  # Cap predictions at 10% above max value of y_test

# # %%
# visualization(datetime1, y_pred, y_test, 3000,3300)
# print_model_weights_and_biases(model)

# # %%
# df, X, Y = CAMELSrun('13340000', 3, 7, 'Flow+2', False)

# # %%
# predictions1, datetime2 = backtest(df, model, X, Y, 365, 60, None)

# y_pred1 = predictions1["prediction"]
# y_test1 = predictions1["actual"]

# y_pred1 = np.maximum(y_pred1, 0)  # Ensure no negative predictions1
# y_pred1 = np.minimum(y_pred1, np.max(y_test1) * 1.1)  # Cap predictions1 at 10% above max value of y_test

# # %%
# visualization(datetime2, y_pred1, y_test1, 3400, 3700)
# print_model_weights_and_biases(model)

# # %%
# df, X, Y = CAMELSrun('05131500', 3, 7, 'Flow+2', False)

# # %%
# predictions2, datetime3 = backtest(df, model, X, Y, 540, 60, pd.Timestamp('2004-01-01'))

# y_pred2 = predictions2["prediction"]
# y_test2 = predictions2["actual"]

# y_pred2 = np.maximum(y_pred2, 0)  # Ensure no negative predictions2
# y_pred2 = np.minimum(y_pred2, np.max(y_test2) * 1.1)  # Cap predictions2 at 10% above max value of y_test


# # %%
# visualization(datetime3, y_pred2, y_test2, 0, 200)
# print_model_weights_and_biases(model)

# # %%
# # def montecarlo(df, X, Y, num_samples=100, num_steps=7):

# #     # num_steps: Number of future time steps to predict 
# #     # num_samples: Number of Monte Carlo samples to draw

# #     scaler_X = StandardScaler()
# #     scaler_Y = StandardScaler()

# #     model.trainable = True
# #     all_predictions = []




# #     if isinstance(Y, pd.Series):
# #         Y_column = Y.name
# #     else:
# #         Y_column = Y

# #     test_indices = []

# #     # Fitting???

# #     samples = df.iloc[-num_samples:, :]
# #     X_samples = samples[X]
# #     X_scaled = scaler_X.fit_transform(X_samples)
# #     Y_samples = samples[[Y_column]]
# #     Y_scaled = scaler_Y.fit_transform(Y_samples).reshape(-1, 1)


# #     for _ in range(num_samples):
# #         # Generate future predictions
# #         predictions = []
# #         lastX = X_scaled[-1]  # Start with the last known scaled input data

# #         for _ in range(num_steps):
# #             pred = model.predict(lastX, verbose=0)
# #             pred = scaler_Y.inverse_transform(pred).reshape(-1, 1).flatten()    # Inverse scale the prediction
# #             predictions.append(pred)

# #             lastX = np.roll(lastX, -1)  # Shift the input data to simulate time step
# #             lastX[-1] = pred

# #         all_predictions.append(np.array(predictions).flatten())


# #     # Convert to numpy array for easy manipulation
# #     all_predictions = np.array(all_predictions)

# #     # Calculate mean and uncertainty bounds
# #     mean_prediction = np.mean(all_predictions, axis=0)
# #     lower_bound = np.percentile(all_predictions, 5, axis=0)
# #     upper_bound = np.percentile(all_predictions, 95, axis=0)

# #     # Inverse scale the original last 100 Y values
# #     Y_original_scaled = scaler_Y.inverse_transform(Y_scaled).reshape(-1, 1).flatten()

# #     # Plotting
# #     plt.plot(range(100), Y_original_scaled, label='Original Data', color='blue')
# #     plt.plot(range(100, 130), mean_prediction, label='Forecasted Trend', color='orange')
# #     plt.fill_between(range(100, 130), lower_bound, upper_bound, color='gray', alpha=0.5, label='95% Confidence Interval')
# #     plt.legend()
# #     plt.show()

# # montecarlo(df, X, Y)


