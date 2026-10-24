import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load dataset and create output directory
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid')

# 1. Product returns category-wise (Number and Percentage separately)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
cat_returns = df[df['returned'] == 1].groupby('product_category')[
    'order_id'
].count()
cat_total = df.groupby('product_category')['order_id'].count()
cat_return_rate = (cat_returns / cat_total) * 100

cat_returns.sort_values(ascending=False).plot(
    kind='bar', ax=ax1, color='skyblue'
)
ax1.set_title('Total Returns by Category (Count)')
ax1.set_ylabel('Number of Returns')
ax1.set_xlabel('Product Category')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

cat_return_rate.sort_values(ascending=False).plot(
    kind='bar', ax=ax2, color='salmon'
)
ax2.set_title('Return Rate by Category (%)')
ax2.set_ylabel('Return Rate (%)')
ax2.set_xlabel('Product Category')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('plots/1_category_returns.png', dpi=300)
plt.close()

# 2. Customers with past returns
plt.figure(figsize=(8, 5))
past_returns = (
    df.groupby('previous_returns_count')['returned'].mean().reset_index()
)
past_returns['return_rate_pct'] = past_returns['returned'] * 100

sns.barplot(
    x='previous_returns_count',
    y='return_rate_pct',
    data=past_returns,
    palette='crest',
    hue='previous_returns_count',
    dodge=False,
)
plt.title('Return Rate by Customer Previous Returns Count')
plt.xlabel('Previous Returns Count')
plt.ylabel('Return Rate (%)')
plt.legend([], [], frameon=False)
plt.savefig('plots/2_customer_past_returns.png', dpi=300)
plt.close()

print('Customer past returns graph saved successfully!')

# 3. Review rating and return
plt.figure(figsize=(8, 5))
review_return = df.groupby('review_rating')['returned'].mean().reset_index()
review_return['return_rate_pct'] = review_return['returned'] * 100
sns.barplot(x='review_rating', y='return_rate_pct', data=review_return, palette='viridis', hue='review_rating', dodge=False)
plt.title('Return Rate by Review Rating')
plt.xlabel('Review Rating')
plt.ylabel('Return Rate (%)')
plt.legend([], [], frameon=False)
plt.savefig('plots/3_review_rating_return.png', dpi=300)
plt.close()

# 4. Discount and return
plt.figure(figsize=(8, 5))
df['discount_group'] = pd.cut(df['discount_pct'], bins=[-1, 0, 10, 20, 30, 50, 100], labels=['0%', '1-10%', '11-20%', '21-30%', '31-50%', '50%+'])
discount_return = df.groupby('discount_group', observed=False)['returned'].mean().reset_index()
discount_return['return_rate_pct'] = discount_return['returned'] * 100
sns.barplot(x='discount_group', y='return_rate_pct', data=discount_return, palette='magma', hue='discount_group', dodge=False)
plt.title('Return Rate by Discount Percentage')
plt.xlabel('Discount Range')
plt.ylabel('Return Rate (%)')
plt.legend([], [], frameon=False)
plt.savefig('plots/4_discount_return.png', dpi=300)
plt.close()

print("Both charts successfully updated to clean bar charts and saved!")
# 5. Slow delivery and return
plt.figure(figsize=(8, 5))
delivery_return = (
    df.groupby('delivery_days')['returned'].mean().reset_index()
)
delivery_return['return_rate_pct'] = delivery_return['returned'] * 100
sns.lineplot(
    x='delivery_days',
    y='return_rate_pct',
    data=delivery_return,
    marker='o',
    color='b',
    linewidth=2,
)
plt.title('Return Rate vs Delivery Days (Slow Delivery Analysis)')
plt.xlabel('Delivery Days')
plt.ylabel('Return Rate (%)')
plt.savefig('plots/5_slow_delivery_return.png', dpi=300)
plt.close()

# 6. Rating of seller and return
plt.figure(figsize=(8, 5))
df['seller_rating_bin'] = pd.cut(
    df['seller_rating'], bins=[0, 2, 3, 4, 5], labels=['<2', '2-3', '3-4', '4-5']
)
seller_return = (
    df.groupby('seller_rating_bin', observed=False)['returned']
    .mean()
    .reset_index()
)
seller_return['return_rate_pct'] = seller_return['returned'] * 100
sns.barplot(
    x='seller_rating_bin',
    y='return_rate_pct',
    data=seller_return,
    palette='crest',
)
plt.title('Return Rate by Seller Rating')
plt.xlabel('Seller Rating Range')
plt.ylabel('Return Rate (%)')
plt.savefig('plots/6_seller_rating_return.png', dpi=300)
plt.close()

# 7. Return rate as price go up
plt.figure(figsize=(10, 5))
df['price_bin'] = pd.qcut(df['price'], q=10)
price_return = (
    df.groupby('price_bin', observed=False)['returned'].mean().reset_index()
)
price_return['return_rate_pct'] = price_return['returned'] * 100
price_return['price_mid'] = price_return['price_bin'].apply(lambda x: x.mid)
sns.lineplot(
    x='price_mid',
    y='return_rate_pct',
    data=price_return,
    marker='s',
    color='r',
    linewidth=2,
)
plt.title('Return Rate as Product Price Increases (Deciles)')
plt.xlabel('Product Price (Midpoint of Decile)')
plt.ylabel('Return Rate (%)')
plt.savefig('plots/7_price_return.png', dpi=300)
plt.close()

# 8. Monthly return rate
plt.figure(figsize=(12, 5))
df['order_month'] = pd.to_datetime(df['order_datetime']).dt.to_period('M')
monthly_return = (
    df.groupby('order_month')['returned'].agg(['count', 'mean']).reset_index()
)
monthly_return['order_month_str'] = monthly_return['order_month'].astype(str)
monthly_return['return_rate_pct'] = monthly_return['mean'] * 100
sns.lineplot(
    x='order_month_str',
    y='return_rate_pct',
    data=monthly_return,
    marker='o',
    color='purple',
    linewidth=2,
)
plt.title('Monthly Return Rate Trend Over Time')
plt.xlabel('Order Month')
plt.ylabel('Return Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/8_monthly_return.png', dpi=300)
plt.close()

# 9. Why products are returned
plt.figure(figsize=(10, 5))
reason_counts = df[df['returned'] == 1]['return_reason'].value_counts()
sns.barplot(x=reason_counts.values, y=reason_counts.index, palette='Spectral')
plt.title('Reasons for Product Returns')
plt.xlabel('Number of Returns')
plt.ylabel('Return Reason')
plt.savefig('plots/9_return_reasons.png', dpi=300)
plt.close()

print('All 9 graphs successfully generated and saved to the "plots/" directory!')