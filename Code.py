import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

# Load your dataset (replace with actual file path or dataframe)
# Assumed columns: 'customer_id', 'merchant_id', 'recommendation_type', 'purchase_made', <feature columns>
data = pd.read_csv('amex_transaction_data.csv')

# Split into treated (active recommendation) and control (organic recommendation) groups
treated = data[data['recommendation_type'] == 'active']
control = data[data['recommendation_type'] == 'organic']

# Features and target for treated and control groups
features = [col for col in data.columns if col not in ['customer_id', 'merchant_id', 'recommendation_type', 'purchase_made']]
X_treated = treated[features]
y_treated = treated['purchase_made']

X_control = control[features]
y_control = control['purchase_made']

# Split treated and control groups into training and test sets
X_train_treated, X_test_treated, y_train_treated, y_test_treated = train_test_split(X_treated, y_treated, test_size=0.2, random_state=42)
X_train_control, X_test_control, y_train_control, y_test_control = train_test_split(X_control, y_control, test_size=0.2, random_state=42)

# Train GradientBoostingRegressor for treated group
model_treated = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_treated.fit(X_train_treated, y_train_treated)

# Train GradientBoostingRegressor for control group
model_control = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_control.fit(X_train_control, y_train_control)

# Predict outcomes for treated and control groups on test set
y_pred_treated_treated = model_treated.predict(X_test_treated)
y_pred_control_treated = model_control.predict(X_test_treated)

y_pred_treated_control = model_treated.predict(X_test_control)
y_pred_control_control = model_control.predict(X_test_control)

# Calculate uplift scores
uplift_scores_treated = y_pred_treated_treated - y_pred_control_treated
uplift_scores_control = y_pred_treated_control - y_pred_control_control

# Aggregate results
test_results_treated = pd.DataFrame({
    'customer_id': treated.iloc[X_test_treated.index]['customer_id'],
    'uplift_score': uplift_scores_treated
})

test_results_control = pd.DataFrame({
    'customer_id': control.iloc[X_test_control.index]['customer_id'],
    'uplift_score': uplift_scores_control
})

# Combine results
uplift_results = pd.concat([test_results_treated, test_results_control])

# Sort customers by uplift score (highest uplift first)
uplift_results_sorted = uplift_results.sort_values(by='uplift_score', ascending=False)

# Matching top 10 merchants
top_10_merchants = data.groupby('merchant_id').size().sort_values(ascending=False).head(10).index

# Match customers to top 10 merchants based on uplift score
uplift_results_sorted['matched_merchant'] = uplift_results_sorted['customer_id'].map(
    lambda cid: top_10_merchants[np.random.randint(0, len(top_10_merchants))]  # Randomly assign merchant for demo purposes
)

# Output results
print(uplift_results_sorted.head(10))
