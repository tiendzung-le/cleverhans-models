

import argparse
import pandas as pd
import numpy as np


def run(output_file):
    pred1 = pd.read_csv("/tmp/pred_1.csv", header=None)
    pred1.columns = ["picture", "prob1"]
    pred2 = pd.read_csv("/tmp/pred_2.csv", header=None)
    pred2.columns = ["picture", "prob2"]
    pred3 = pd.read_csv("/tmp/pred_3.csv", header=None)
    pred3.columns = ["picture", "prob3"]
    pred4 = pd.read_csv("/tmp/pred_4.csv", header=None)
    pred4.columns = ["picture", "prob4"]
    pred5 = pd.read_csv("/tmp/pred_5.csv", header=None)
    pred5.columns = ["picture", "prob5"]
    pred6 = pd.read_csv("/tmp/pred_6.csv", header=None)
    pred6.columns = ["picture", "prob6"]
    pred7 = pd.read_csv("/tmp/pred_7.csv", header=None)
    pred7.columns = ["picture", "prob7"]

    pred_fool = pd.read_csv("/tmp/pred_fool.csv", header=None)
    pred_fool.columns = ["picture", "prob_fool"]

    rs = pd.merge(pred1, pred2, on=['picture'])
    rs = pd.merge(rs, pred3, on=['picture'])
    rs = pd.merge(rs, pred4, on=['picture'])
    rs = pd.merge(rs, pred5, on=['picture'])
    rs = pd.merge(rs, pred6, on=['picture'])
    rs = pd.merge(rs, pred7, on=['picture'])
    rs = pd.merge(rs, pred_fool, on=['picture'])

    def is_agreed(main_label, labels):
        nb_votes = 0
        for l in labels:
            if l == main_label:
                nb_votes = nb_votes + 1
        return nb_votes >= 4

    def get_label(row):
        prob1 = row["prob1"].split(" ")
        prob2 = row["prob2"].split(" ")
        prob3 = row["prob3"].split(" ")
        prob4 = row["prob4"].split(" ")
        prob5 = row["prob5"].split(" ")
        prob6 = row["prob6"].split(" ")
        prob7 = row["prob7"].split(" ")
        prob_fool = row["prob_fool"].split(" ")

        prob1 = np.array([float(e) for e in prob1])
        prob2 = np.array([float(e) for e in prob2])
        prob3 = np.array([float(e) for e in prob3])
        prob4 = np.array([float(e) for e in prob4])
        prob5 = np.array([float(e) for e in prob5])
        prob6 = np.array([float(e) for e in prob6])
        prob7 = np.array([float(e) for e in prob7])

        prob_fool = np.array([float(e) for e in prob_fool])

        prob = (prob1 + prob2 + prob3 + prob4 + prob5 + prob6 + prob7) / 7.0

        main_label = np.argmax(prob)
        fool_label = np.argmax(prob_fool)

        label = main_label
        if main_label == fool_label: # 80% wrong
            if not is_agreed(main_label,
                         [np.argmax(prob1), np.argmax(prob2), np.argmax(prob3), np.argmax(prob4),
                          np.argmax(prob5), np.argmax(prob6), np.argmax(prob7)]):
                ordered_labels = prob.argsort()[-2:][::-1]
                label = ordered_labels[1] # The second label
        return label

    rs['label'] = rs.apply(lambda x: get_label(x), axis=1)
    rs[["picture", "label"]].to_csv(output_file, index=None, header=None)


def main():
    parser = argparse.ArgumentParser(description='Merge 7 prediction files')

    # define command line arguments
    parser.add_argument('--output_file', dest='output_file', type=str,
                        help='The output file path', required=True)

    # parse the arguments
    args = vars(parser.parse_args())
    output_file = args.get('output_file')

    # execute
    run(output_file)


if __name__ == '__main__':
    main()
