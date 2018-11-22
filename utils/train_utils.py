import pandas as pd
from IPython.display import display
from sklearn.metrics import confusion_matrix


def visualize_results_acc_cm(y_true, y_pred):

    results = pd.DataFrame(data=confusion_matrix(y_true, y_pred),
                           index=['model_0', 'model_1'], columns=['true_0', 'true_1'])
    results_p = results/results.sum().sum()*100

    print("Tulokset määrissä:")
    display(results)
    print("Tulokset prosenteissa:")
    display(results_p)
    print(
        f'Mallin tarkkuus (ON+OP): {results_p.iloc[0,0]+results_p.iloc[1,1]:.3f}%')
