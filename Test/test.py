import random
import csv
predictions_SVM = [random.random() for i in range(50)]
predictions_SVM = zip(range(50), predictions_SVM)

with open("test.csv","w") as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)