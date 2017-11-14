import pandas as pd
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

all_patient_data = pd.read_csv("allPatients.csv")
classes = list(all_patient_data['Classes'])
all_patient_data = all_patient_data.drop('Classes', axis=1)
all_patient_data.replace(to_replace="?", value="NaN")
avg = all_patient_data.values.mean()
all_patient_data.replace(to_replace="NaN", value="avg")

clf = RandomForestClassifier(n_estimators=200, criterion='entropy')

clf = clf.fit(all_patient_data, classes)

genes_dict = dict()
for tree in clf.estimators_:
    i=0
    for feature in tree.feature_importances_:
        if feature > 0:
            if i in genes_dict.keys():
                genes_dict[i] = genes_dict[i] + 1
            else:
                genes_dict[i] = 1
        i=i+1

genes_dict_sorted = sorted(genes_dict, key=genes_dict.get, reverse=True)
most_frequent = genes_dict_sorted[:200]

filtered_genes = []

gene_names = list(all_patient_data.columns.values)
for i in most_frequent:
    filtered_genes.append(gene_names[i])

new_feature_set = pd.DataFrame([all_patient_data[x] for x in all_patient_data if x in filtered_genes]).T

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(new_feature_set)
poly_feature_names = poly.get_feature_names(all_patient_data.columns)

poly_feature_set = pd.DataFrame(poly_features, columns=poly_feature_names)
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(poly_features, classes)
w = clf.coef_[0]

coefs = {}
i = 0
for c in clf.coef_:
    for feat in c:
        coefs[i] = feat
        i = i+1

sorted_genes = sorted(coefs, key=coefs.get, reverse=True)[:200]

filtered_genes = []
cols = list(poly_feature_set.columns.values)
for key in sorted_genes:
    filtered_genes.append(cols[key])

df = pd.DataFrame(filtered_genes, columns=["top200genes"])
df.to_csv('prob3b.csv', index=False)