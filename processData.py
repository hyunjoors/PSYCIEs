import numpy as np
import pandas as pd
import json
import warnings
import operator
import re
import csv
from collections import OrderedDict
from collections import Counter
from operator import itemgetter


class processData:

    def __init__(self, folderPath, filePath):
        self.folderPath = folderPath
        self.filePath = filePath

    def count(self):
        data_train = pd.read_csv(self.filePath)

        respondant_id = data_train.iloc[:, 1:1]
        X = data_train.iloc[:, 1:6]
        y = data_train.iloc[:, 6:11]

        # rename columns for easier access
        X.rename(columns={'open_ended_1': 'A',
                            'open_ended_2': 'C',
                            'open_ended_3': 'E',
                            'open_ended_4': 'N',
                            'open_ended_5': 'O'}, inplace=True)

        y.rename(columns={'E_Scale_score': 'E',
                            'A_Scale_score': 'A',
                            'O_Scale_score': 'O',
                            'C_Scale_score': 'C',
                            'N_Scale_score': 'N'}, inplace=True)
            
        X_group = X.stack().groupby(level=0).apply(' '.join)

        X = X.get_values()
        X_group = X_group.get_values()

        # Hash answers
        (n,m) = X.shape

        # delim = " ", ".", ",", "(", ")", "\n", "\r", "\t", "\b", '\x00', "?"
        # regexPattern = '|'.join(map(re.escape, delim))

        parsed_user_answer_list = []
        for i in range(n):
            # for each user
            user_list = []
            for j in range(5):
                # for each answer
                current_ans = pd.Series(X[i][j])
                current_ans.str.split(pat = ' .,()\n\r\t\b\x00')
                current_ans.str.lower()
                #print(current_ans)
                
                word_count_list = Counter(current_ans) # Update counter with words
                word_count_list = dict(sorted(word_count_list.items(), key = itemgetter(1), reverse = True))
                user_list.append(word_count_list)

            parsed_user_answer_list.append(user_list)
        
        parsed_user_grouped_answer_list = []
        for i in range(n):
            current_ans = pd.Series(X[i][j])
            current_ans.str.split(pat = ' .,()\n\r\t\b\x00')
            current_ans.str.lower()
            print(current_ans)

            word_count_list = Counter(current_ans)
            word_count_list = dict(sorted(word_count_list.items(), key = itemgetter(1), reverse = True))
            parsed_user_grouped_answer_list.append(word_count_list)
        
        ocean = ["A", "C", "E", "N", "O"]
        csvFile = self.folderPath + '/individual_Count.csv'
        with open(csvFile, 'w') as output_file:
            for i in range(n):
                for j in range(5):
                    output_file.write("%s," % ocean[j])
                    for key in parsed_user_answer_list[i][j].keys():
                        output_file.write("%s,%d," % (
                            key, parsed_user_answer_list[i][j][key]))
                    output_file.write("\n")
                output_file.write("\n")

        csvFile = self.folderPath + '/grouped_Counts.csv'
        with open(csvFile, 'w') as output_file:
            for i in range(n):
                for key in parsed_user_grouped_answer_list[i].keys():
                    output_file.write("%s,%d," % (key, parsed_user_grouped_answer_list[i][key]))
                output_file.write("\n")
        return parsed_user_answer_list, parsed_user_grouped_answer_list
