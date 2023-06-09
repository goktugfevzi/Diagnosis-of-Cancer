import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_df = pd.read_csv('data.csv')
labels_df = pd.read_csv('labels.csv',delimiter=",")
target_labels = labels_df['disease_type']

print(data_df.shape)
print(labels_df.shape)

labels_df.head()
labels_df['disease_type'].hist()
labels_df['disease_type'].value_counts()
labels_df.drop(columns=["Sample"], inplace=True)

data_df.drop(columns=["Unnamed: 0"],inplace=True)
data = pd.concat([labels_df, data_df], axis=1, sort=False)
data.dropna(inplace = True)

X = data.drop(['disease_type'], axis=1)
y = data['disease_type']
X_encoded = pd.get_dummies(X, prefix_sep="_")
y_encoded = LabelEncoder().fit_transform(y)
X_scaled = StandardScaler().fit_transform(X_encoded)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size = 0.3, random_state = 150)

model = RandomForestClassifier(n_estimators=700).fit(X_train,y_train)
preds = model.predict(X_test)
print(classification_report(y_test,preds))

feature_imp = pd.Series(model.feature_importances_, index = X_encoded.columns)
feature_imp.nlargest(20).plot(kind="barh")
best_feat = feature_imp.nlargest(10).index

X_reduced = X_encoded[best_feat]
X_reduced.head(10)
Xr_scaled = StandardScaler().fit_transform(X_reduced)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr_scaled,
            y_encoded, test_size = 0.3, random_state = 150)

rmodel = RandomForestClassifier(n_estimators=700).fit(Xr_train,yr_train)
rpreds = rmodel.predict(Xr_test)
print(classification_report(y_test,rpreds))

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(Xr_scaled, y_encoded)
predictions = gbm.predict(Xr_test)
print(classification_report(y_test,predictions))

cm = confusion_matrix(y_test,predictions)
sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,2])
print('Sensitivity1 : ', sensitivity1 )
sensitivity2 = cm[1,1]/(cm[1,1]+cm[1,3])
print('Sensitivity2 : ', sensitivity2 )
sensitivity3 = cm[2,2]/(cm[2,2]+cm[2,0])
print('Sensitivity3 : ', sensitivity3 )
sensitivity4 = cm[3,3]/(cm[3,3]+cm[3,1])
print('Sensitivity4 : ', sensitivity4)