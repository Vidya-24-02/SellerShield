"""
SellerShield — train_model.py
Run this ONCE before starting the app:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

np.random.seed(42)
N = 50000

print("🔧 Generating 50,000 seller profiles...")

PLATFORMS = ['Amazon','Flipkart','Meesho','eBay','Etsy','Shopee','Myntra','Snapdeal']
platform_fraud = {'Amazon':0.12,'Flipkart':0.15,'Meesho':0.28,'eBay':0.20,
                  'Etsy':0.10,'Shopee':0.25,'Myntra':0.13,'Snapdeal':0.22}

data = {}
data['platform'] = np.random.choice(PLATFORMS, N)
base_fraud = np.array([platform_fraud[p] for p in data['platform']])

data['account_age_months'] = np.where(np.random.rand(N)<base_fraud,
    np.random.exponential(4,N).clip(1,24), np.random.exponential(30,N).clip(3,120))
data['total_reviews'] = np.where(np.random.rand(N)<base_fraud,
    np.random.exponential(25,N).clip(0,200).astype(int), np.random.exponential(400,N).clip(10,5000).astype(int))
data['avg_rating'] = np.where(np.random.rand(N)<base_fraud,
    np.clip(np.random.normal(4.7,0.2,N),1,5), np.clip(np.random.normal(4.1,0.5,N),1,5))
data['rating_std'] = np.where(np.random.rand(N)<base_fraud,
    np.abs(np.random.normal(0.1,0.05,N)), np.abs(np.random.normal(0.8,0.3,N)))
data['return_rate'] = np.where(np.random.rand(N)<base_fraud,
    np.clip(np.random.beta(5,1.5,N),0,1), np.clip(np.random.beta(1.5,8,N),0,1))
data['response_time_hrs'] = np.where(np.random.rand(N)<base_fraud,
    np.clip(np.random.exponential(48,N),1,200), np.clip(np.random.exponential(6,N),0.5,72))
data['price_deviation_pct'] = np.where(np.random.rand(N)<base_fraud,
    np.clip(np.random.normal(-35,15,N),-80,0), np.clip(np.random.normal(0,12,N),-20,40))
data['platform_verified'] = np.where(np.random.rand(N)<base_fraud,
    np.random.binomial(1,0.08,N), np.random.binomial(1,0.65,N))
data['listing_quality'] = np.where(np.random.rand(N)<base_fraud,
    np.clip(np.random.normal(38,18,N),5,100), np.clip(np.random.normal(72,18,N),10,100))
data['dispute_rate'] = np.where(np.random.rand(N)<base_fraud,
    np.clip(np.random.beta(4,2,N),0,1), np.clip(np.random.beta(1,8,N),0,1))
data['repeat_buyer_rate'] = np.where(np.random.rand(N)<base_fraud,
    np.clip(np.random.beta(1,5,N),0,1), np.clip(np.random.beta(4,3,N),0,1))
data['keyword_risk_score'] = np.where(np.random.rand(N)<base_fraud,
    np.clip(np.random.normal(65,20,N),0,100), np.clip(np.random.normal(25,18,N),0,100))

df = pd.DataFrame(data)
df_enc = pd.get_dummies(df, columns=['platform'], prefix='platform')

# Labels
fraud_prob = (
    (df['account_age_months']<6).astype(float)*0.25 +
    (df['total_reviews']<20).astype(float)*0.20 +
    (df['rating_std']<0.2).astype(float)*0.15 +
    (df['return_rate']>0.4).astype(float)*0.12 +
    (df['price_deviation_pct']<-30).astype(float)*0.10 +
    (df['dispute_rate']>0.35).astype(float)*0.10 +
    (df['platform_verified']==0).astype(float)*0.08 +
    base_fraud*0.5
).clip(0,1)

labels = []
for fp in fraud_prob:
    r = np.random.rand()
    if fp>0.55 or (fp>0.4 and r<0.6): labels.append('High Risk')
    elif fp>0.25 or (fp>0.15 and r<0.5): labels.append('Moderate Risk')
    else: labels.append('Trusted')
df_enc['label'] = labels

feature_cols = [c for c in df_enc.columns if c != 'label']
X = df_enc[feature_cols]
y = df_enc['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Dataset ready — Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Labels: {dict(pd.Series(labels).value_counts())}\n")

print("🌳 Training Random Forest (200 trees)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_split=4,
    min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)

print("⚡ Training Gradient Boosting (150 trees)...")
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
    max_depth=6, subsample=0.8, random_state=42)

print("🤝 Building Voting Ensemble...")
ensemble = VotingClassifier(estimators=[('rf',rf),('gb',gb)], voting='soft', weights=[2,1])
pipeline = Pipeline([('scaler', StandardScaler()), ('model', ensemble)])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv = cross_val_score(pipeline, X_train[:5000], y_train[:5000], cv=5, scoring='accuracy')

print(f"\n📊 Test Accuracy : {acc*100:.2f}%")
print(f"   CV Accuracy   : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred)}")

os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/seller_model.pkl')
joblib.dump(feature_cols, 'model/feature_names.pkl')
joblib.dump(list(pipeline.classes_) if hasattr(pipeline,'classes_') else ['High Risk','Moderate Risk','Trusted'], 'model/classes.pkl')

# Save classes properly
classes = list(pipeline.named_steps['model'].classes_)
joblib.dump(classes, 'model/classes.pkl')

with open('model/report.txt','w') as f:
    f.write(f"Accuracy: {acc*100:.2f}%\nCV: {cv.mean()*100:.2f}%\n\n{classification_report(y_test,y_pred)}")

print("\n💾 Saved: model/seller_model.pkl, model/feature_names.pkl, model/classes.pkl")
print("🚀 Now run: python server.py")
