import csv


class Evaluator:
    def __init__(self, result, raw_data):
        self.result = result
        self.train_edges = raw_data.train_edges
        self.pred_edges = raw_data.pred_edges

    def write_result(self, result):
        file_path = "C:\\users\\73639\desktop\\Result\\"
        import datetime
        file_name = datetime.datetime.now().strftime('%d_%H_%M_%S')
        out = open(file_path + file_name + 'out.csv', 'a', encoding='utf-8', newline='')
        csv_writer = csv.writer(out, dialect='excel')
        csv_writer.writerow(['Id', 'Predicted'])
        for id, r in result:
            csv_writer.writerow([id, r])

    def combine_result(self, pred_Y):
        result = []
        assert len(pred_Y) == 2000
        for i in range(2000):
            result.append([str(i + 1), pred_Y[i]])
        return result

    def evaluate(self):
        t = 0
        result = self.result
        train_edges = self.train_edges
        pred_edges = self.pred_edges
        result_rank = sorted(result)

        posi_count = 0
        nega_count = 0
        for i, r in enumerate(result):
            if pred_edges[i] in train_edges:
                if result_rank.index(result[i]) < 1000:
                    t += 1
                    result[i] = 1
            if result[i] > 0.5:
                posi_count += 1
            else:
                nega_count += 1

        print('failed edge', t)
        print('Prediction finished')
        print('PC', posi_count)
        print('NC', nega_count)
        print()
        print('Writing to file')
        result = self.combine_result(result)
        self.write_result(result)
        print('File generated')

