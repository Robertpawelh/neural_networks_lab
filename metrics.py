def accuracy_score(Y_pred, Y_test):
    accuracy_score = (Y_pred == Y_test).sum() / len(Y_pred)
    return accuracy_score

def classification_error(Y_pred, Y_test):
    classification_error = (len(Y_test) - (Y_pred == Y_test).sum()) / len(Y_test)
    return classification_error
