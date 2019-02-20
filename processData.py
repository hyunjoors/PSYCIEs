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

from sklearn.neighbors import KNeighborsRegressor


class processData:

    def count(folderPath, filePath):
        data_train = pd.read_csv(filePath)

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

        delim = " ", ".", ",", "(", ")", "\n", "\r", "\t", "\b", '\x00', "?"
        regexPattern = '|'.join(map(re.escape, delim))

        parsed_user_answer_list = []
        for i in range(n):
            # for each user
            user_list = []
            for j in range(5):
                # for each answer
                word_list = re.split(regexPattern, X[i][j])
                word_list = list(filter(None, word_list))
                word_list = [x.lower() for x in word_list]
                sentence_dict = Counter()
                sentence_dict.update(word_list) # Update counter with words
                sentence_dict = dict(sorted(sentence_dict.items(), key = itemgetter(1), reverse = True))
                user_list.append(sentence_dict)

            parsed_user_answer_list.append(user_list)
        
        ocean = ["A", "C", "E", "N", "O"]
        csvFile = folderPath + '/individual_Count.csv' 
        with open(csvFile, 'w') as output_file:
            for i in range(n):
                for j in range(5):
                    output_file.write("%s," % ocean[j])
                    for key in parsed_user_answer_list[i][j].keys():
                        output_file.write("%s,%d," % (key, parsed_user_answer_list[i][j][key]))
                    output_file.write("\n")
                output_file.write("\n")

        parsed_user_grouped_answer_list = []
        for i in range(n):
            word_list = re.split(regexPattern, X_group[i])
            word_list = list(filter(None, word_list))
            word_list = [x.lower() for x in word_list]
            sentence_dict = Counter()
            sentence_dict.update(word_list) 
            sentence_dict = dict(sorted(sentence_dict.items(), key = itemgetter(1), reverse = True))
            parsed_user_grouped_answer_list.append(sentence_dict)
        
        csvFile = folderPath + '/grouped_Counts.csv' 
        with open(csvFile, 'w') as output_file:
            for i in range(n):
                for key in parsed_user_grouped_answer_list[i].keys():
                    output_file.write("%s,%d," % (key, parsed_user_grouped_answer_list[i][key]))
                output_file.write("\n")
        return parsed_user_answer_list, parsed_user_grouped_answer_list


    def summarize(dictionaryType):
        for i in range(5):
            print("blah")
        return
