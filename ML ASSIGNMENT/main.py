import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, davies_bouldin_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# loading the data into program
df = pd.read_excel('data.xlsx')
X = df.drop(columns=['order_id', 'customer_id', 'product_id', 'order_datetime',
                     'return_reason', 'returned'])
y = df['returned']
X = pd.get_dummies(X, columns=['product_category', 'shipping_type',
                               'payment_method', 'order_weekday'], drop_first=True)

# ---------- clustering ----------
num = ['price', 'quantity', 'delivery_days', 'seller_rating', 'previous_returns_count',
       'customer_tenure_days', 'customer_total_orders', 'discount_pct', 'review_rating']
Xc = StandardScaler().fit_transform(X[num])

print(' CUSTOMER SEGMENTS ')
# try k = 2 to 6 and compare scores
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(Xc)
    print(f'K={k} | inertia {km.inertia_:.1f} | silhouette {silhouette_score(Xc, lab):.4f} '
          f'| DB {davies_bouldin_score(Xc, lab):.4f}')

# K=4 looked best so going with that
df['Behavior_Cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(Xc)
print('\nCluster profiles:\n', df.groupby('Behavior_Cluster')[num].mean())

# ---------- classification ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# scale only the numeric columns (fit on train only)
scale_cols = num + ['order_hour', 'return_score']
sc = StandardScaler()
X_train[scale_cols] = sc.fit_transform(X_train[scale_cols])
X_test[scale_cols] = sc.transform(X_test[scale_cols])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
}

print('\n=== RETURN RISK MODELS ===')
rows = []
plt.figure(figsize=(8, 6))
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    prob = m.predict_proba(X_test)[:, 1]
    rep = classification_report(y_test, pred, output_dict=True)['1']
    auc = roc_auc_score(y_test, prob)
    rows.append({'Model': name, 'Accuracy': (pred == y_test).mean(),
                 'Precision': rep['precision'], 'Recall': rep['recall'],
                 'F1': rep['f1-score'], 'ROC-AUC': auc,
                 'CV F1': cross_val_score(m, X_train, y_train, cv=5, scoring='f1').mean()})
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
    print(f'\n{name}\n', confusion_matrix(y_test, pred))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.savefig('graph', dpi=300)
plt.show()

print('\n=== MODEL COMPARISON ===')
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---------- grid search on random forest ----------
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_split': [2, 5]},
                    cv=3, scoring='f1', n_jobs=-1).fit(X_train, y_train)
print('\nBest params:', grid.best_params_)
print('Best CV F1:', round(grid.best_score_, 4))

# ---------- return reasons + lookup ----------
print('\n=== RETURN REASONS ===')
print(df[df['returned'] == 1]['return_reason'].value_counts().to_string())

# which column to filter on for each search type
cols = {'customer': 'customer_id', 'product': 'product_id', 'category': 'product_category'}


def lookup(kind, value):
    if kind not in cols:
        return print("Wrong type, use 'customer', 'product' or 'category'")

    sub = df[df[cols[kind]].str.lower() == value.lower()]
    if sub.empty:
        return print(f"\n{kind} '{value}' not found")

    n, r = len(sub), sub['returned'].sum()
    print(f'\n--- {kind.title()} Profile: {value.upper()} ---')
    if kind == 'product':
        print('Category:', sub['product_category'].iloc[0])
    print(f'Orders: {n} | Returned: {r} | Return rate: {r / n * 100:.2f}%')

    # customers get their full order history, the rest get a reasons breakdown
    if kind == 'customer':
        print(sub[['order_id', 'product_id', 'product_category', 'price',
                   'returned', 'return_reason']].to_string(index=False))
    else:
        reasons = sub[sub['returned'] == 1]['return_reason'].value_counts()
        print('\nReturn reasons:')
        print(reasons.to_string() if not reasons.empty else 'No returns yet.')


print("\n=== RETURN LOOKUP SYSTEM (type 'exit' to quit) ===")
while True:
    kind = input("Search by (customer / product / category): ").strip().lower()
    if kind == 'exit':
        break
    value = input("Enter ID or name (e.g. C11772, P21604, Electronics): ").strip()
    if value.lower() == 'exit':
        break
    lookup(kind, value)
    print('\n' + '=' * 50 + '\n')