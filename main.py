import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. here we load the data
df = pd.read_excel('data.xlsx')

# 
numeric_features = [
    'price', 'quantity', 'delivery_days', 'seller_rating', 
    'previous_returns_count', 'customer_tenure_days', 
    'customer_total_orders', 'discount_pct', 'review_rating', 
    'order_hour', 'return_score'
]

categorical_features = [
    'product_category', 'shipping_type', 
    'payment_method', 'order_weekday'
]

# 3. 
drop_cols = ['order_id', 'customer_id', 'product_id', 'order_datetime', 'return_reason', 'returned']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['returned']

# 4. One hot encoder
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# define which columns describe customer behavior for clustering
num = [
    'price',
    'quantity',
    'delivery_days',
    'seller_rating',
    'previous_returns_count',
    'customer_tenure_days',
    'customer_total_orders',
    'discount_pct',
    'review_rating',
]

# scale the numerical features so they have equal weight in kmeans
Xc = StandardScaler().fit_transform(X[num])

print('--- TESTING K VALUES FOR CUSTOMER SEGMENTS ---')
# loop through k values from 2 to 6 to see how well the clusters form
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(Xc)

    # higher silhouette is better, lower davies-bouldin is better
    sil = silhouette_score(Xc, lab)
    db = davies_bouldin_score(Xc, lab)

    print(f'K={k} | inertia: {km.inertia_:.1f} | silhouette: {sil:.4f} | DB: {db:.4f}')

# k=4 gave the best separation, so we lock it in and save labels back to the dataframe
df['Behavior_Cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(Xc)

print('\nAverage values for each customer behavior cluster:')
print(df.groupby('Behavior_Cluster')[num].mean())




import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. Split data into training and testing sets (stratified to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Scale numeric features (fit on train only to prevent data leakage)
scale_cols = num + ['order_hour', 'return_score']
scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# 3. Define our classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
}

# 4. Train each model, evaluate metrics, and plot ROC curves
print('\n=== TRAINING CLASSIFICATION MODELS ===')
results = []
plt.figure(figsize=(8, 6))

for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    prob = m.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, pred, output_dict=True)['1']
    auc = roc_auc_score(y_test, prob)
    cv_f1 = cross_val_score(m, X_train, y_train, cv=5, scoring='f1').mean()

    results.append(
        {
            'Model': name,
            'Accuracy': (pred == y_test).mean(),
            'Precision': report['precision'],
            'Recall': report['recall'],
            'F1': report['f1-score'],
            'ROC-AUC': auc,
            'CV F1': cv_f1,
        }
    )

    # Plot ROC curve for comparison
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
    
    print(f'\nConfusion Matrix for {name}:\n', confusion_matrix(y_test, pred))

# Finalize and save the ROC plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.savefig('roc_plot.png', dpi=300)
plt.show()

# Print out a clean comparison table of all models
print('\n=== MODEL COMPARISON TABLE ===')
print(pd.DataFrame(results).round(4).to_string(index=False))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# X_train, y_train and df come from the earlier part of the notebook

# --- tuning the random forest ---
print("\nTuning random forest...")

params = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), params,
                    cv=3, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV f1:", round(grid.best_score_, 4))


# --- return lookup ---
print("\nReturn reasons overall:")
returned_df = df[df["returned"] == 1]
print(returned_df["return_reason"].value_counts())

# maps what the user types to the column we search in
columns = {
    "customer": "customer_id",
    "product": "product_id",
    "category": "product_category",
}

print("\nReturn lookup (type exit to quit)")

while True:
    kind = input("Search by customer / product / category: ").strip().lower()
    if kind == "exit":
        break
    if kind not in columns:
        print("Please type customer, product or category.")
        continue

    value = input("Enter the id or category EX- CUSTOMER ID:C12194 ,C11772 , PRODUCT ID:P21679,P22471 , ORDER ID:O100695,CATEGORY :Beauty,Clothing  ").strip()
    if value.lower() == "exit":
        break

    col = columns[kind]
    rows = df[df[col].str.lower() == value.lower()]

    if len(rows) == 0:
        print("Nothing found for", kind, value, "- try again.")
        continue

    orders = len(rows)
    returns = rows["returned"].sum()
    rate = returns / orders * 100

    print("\n" + kind.upper(), "report for", value.upper())
    if kind == "product":
        print("Category:", rows["product_category"].iloc[0])
    print("Orders:", orders, "| Returns:", returns, "| Return rate: %.2f%%" % rate)

    if kind == "customer":
        print("\nOrder history:")
        show = ["order_id", "product_id", "product_category",
                "price", "returned", "return_reason"]
        print(rows[show].to_string(index=False))
    else:
        reasons = rows[rows["returned"] == 1]["return_reason"].value_counts()
        print("\nWhy items were returned:")
        if reasons.empty:
            print("No returns yet.")
        else:
            print(reasons)

    print("\n" + "-" * 40 + "\n")