import pandas as pd
from sklearn import svm

''' Get most important genes in relation to Alzheimer's expression'''
all_patient_data = pd.read_csv("allPatients.csv")
classes = list(all_patient_data['Classes'])
all_patient_data = all_patient_data.drop('Classes', axis=1)
all_patient_data.replace(to_replace="?", value="NaN")
avg = all_patient_data.values.mean()
all_patient_data.replace(to_replace="NaN", value="avg")

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(all_patient_data, classes)
w = clf.coef_[0]

coefs = {}
i = 0
for c in clf.coef_:
    for feat in c:
        coefs[i] = feat
        i = i+1

sorted_genes = sorted(coefs, key=coefs.get, reverse=True)[:200]

filtered_genes = []
cols = list(all_patient_data.columns.values)
for key in sorted_genes:
    filtered_genes.append(cols[key])

df = pd.DataFrame(filtered_genes, columns=["top200genes"])
# print(df)
df.to_csv('filtered_genes.csv', index=False)
