from classifier import Classifier
from dataframe import DataFrame
from evalue import Evaluator
from raw import RawData

if __name__ == '__main__':
    rd = RawData()
    df = DataFrame(rd)
    result = Classifier(df).get_result()
    # result = [0 for _ in range(2000)]
    Evaluator(result, rd).evaluate()
