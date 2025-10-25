# 如何运行？
python perceptron.py
相同目录放好Data.csv文件

# 结果
~~~
原始标签分布: [   0 1655  295  176]

=== Task 1: Perceptron ===
Perceptron did not converge within 2000 epochs
Perceptron Results:
  Accuracy : 0.8873
  Precision: 0.9034
  Recall   : 0.9578
  F1-score : 0.9298

Classification Report:
                precision    recall  f1-score   support

Abnormal (2,3)       0.81      0.64      0.71        94
    Normal (1)       0.90      0.96      0.93       332

      accuracy                           0.89       426
     macro avg       0.86      0.80      0.82       426
  weighted avg       0.88      0.89      0.88       426


=== Task 2: SVM Comparison ===
SVM (linear  ) → Acc: 0.8920, F1: 0.9311
SVM (poly    ) → Acc: 0.9038, F1: 0.9400
SVM (rbf     ) → Acc: 0.9202, F1: 0.9496
SVM (sigmoid ) → Acc: 0.7864, F1: 0.8627

--- Tuning RBF SVM ---
Best RBF SVM → F1: 0.9615, Params: {'C': 10, 'gamma': 'scale'}

Final SVM Performance:
                precision    recall  f1-score   support

Abnormal (2,3)       0.91      0.80      0.85        94
    Normal (1)       0.94      0.98      0.96       332

      accuracy                           0.94       426
     macro avg       0.93      0.89      0.91       426
  weighted avg       0.94      0.94      0.94       426
~~~

