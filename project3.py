import csv
import json
import pandas as pd
import numpy as np
import warnings
import operator
from numpy.linalg import norm
from scipy import spatial

def user_rating(dish_id, user_id, user_ratings):
    # Your code should output the rating of the user for that dish.
    # If the rating doesn't exist, use memory based collaborative filtering
    # to estimate the user's rating for the specified dish.
    # Refer to course slides for details.

    status = ["existing", "estimated"]
    n = len(dishes)-2
    user_given = user_ratings[user_id]

    #if any(user_id in user for user in user_ratings): 
    #    found_user = user_ratings[user_id]

    for dish in user_given:
        # dish was found
        # return the rating of the dish
        if dish_id == dish[0]:
                return dish[1], status[0]
        #else:
    
    # the rating doesn't exist
    # use memory based collaborative filtering to estimate

    # get all users that rated dish_id
    user_id_list = list()
    for user in user_ratings:
        for dish in user_ratings[user]:
            if dish_id == dish[0]:
                user_id_list.append(user)
    # now we have all users who rated dish_id

    # 1. calculate the W(vector space similarity) between the users
    W = dict()
    user_given = dict(user_given)
    for user in user_id_list:
        # user_id_list: the list of user_id that includes dish_id
        # user: the id of a user in the user_id_list
        if user is not user_id:
            dot_sum = 0
            norm1 = 0
            norm2 = 0
            user_temp = dict(user_ratings[user])
            list1 = user_given.keys() # list of dish id in user_given
            list2 = user_temp.keys() # list of dish id in user_ratings[user]

            for dish in list(set(list1).intersection(list2)):
                rate1 = user_given.get(dish)
                rate2 = user_temp.get(dish)
                dot_sum += (rate1 * rate2)
                norm1 += (rate1**2)
                norm2 += (rate2**2)

            if norm1 == 0 or norm2 == 0:
                W[user] = 0
            else:
                W[user] = ( dot_sum / ( np.sqrt(norm1) * np.sqrt(norm2) ) )


    # 2. calulate the avgerage of all users and get submean of the dish_id for each user
    avg = dict()
    submean = dict()
    for user in user_id_list:
        sum_rating = 0
        temp = dict(user_ratings[user])
        for j in temp.values():
            sum_rating += j
        avg[user] = (sum_rating/len(temp))
        # submean = dish rate - user avg
        submean[user] = temp.get(dish_id) - avg.get(user)

    # 3. make prediction
    # the user's average + sum(other user's dish's submean * W of that user) / sum(W of all other users)
    sum_numer = 0
    sum_denom = 0
    for user in user_id_list:
        if user is not user_id:
            sum_numer += (submean.get(user)*W.get(user))
            sum_denom += W.get(user)
    temp = dict(user_ratings[user_id])
    sum_ = 0
    for j in temp.values():
       sum_ += j
    avg_user_id = sum_ / len(temp)
    rate = avg_user_id + sum_numer / sum_denom

    return rate, status[1]
   




def find_dish(user_id, ingredients_list, dishes):
    # You need to find a dish that contains all the ingredients that are input
    # and suggest a dish that the user hasn't rated yet and is likely to enjoy the most
    # (use user's estimated rating, break ties using average actual ratings of the dishes).
    # You may make this more interesting by thinking of collaborative filtering
    # based on ingredients rather than dishes.

    all_ingred_list = dishes[0][2:]
    dishes = dishes[1:]

    # assign the index of all ingredients
    all_ingred_dict = dict()
    index = 0
    for item in all_ingred_list:
        all_ingred_dict[item] = index
        index += 1



    # find the index of given ingredients
    ingred_index_list = list()
    for item in ingredients_list:
        ingred_index_list.append(all_ingred_dict.get(item))



    # find dishes that contains ingredients_list
    dish_list = list()
    for dish_id in range(len(dishes)):
        count = 0
        for i in range(len(ingred_index_list)):
            if dishes[dish_id][ingred_index_list[i]+2] == str(1):
                count += 1
        if count == len(ingred_index_list):
            #print(dishes[dish_id][1])
            dish_list.append(dish_id)
        
    # there is no dish that contains the ingredients
    if len(dish_list) == 0:
        print("\nNo dish with specified ingredients")
        return


    # check if user has rated any of dish_list
    user_given = dict(user_ratings[user_id])
    dish_unrated_list = list(dish_list)
    for item in user_given.keys():
        for dish in dish_unrated_list:
            if item == dish:
                dish_unrated_list.remove(dish)
    #print("unrated dish list: {}".format(dish_unrated_list))



    # if there is more than 1 dish left in the list, break ties with estimated rating
    if len(dish_unrated_list) == 0:
        print("\nNo new dish with specified ingredients")
        print("Your best-rated dish:")

        ratings = dict()
        for item in dish_list:
            rating, status = user_rating(item, user_id, user_ratings)
            ratings[item] = rating

        max_key = 0
        max_val = 0
        for k, v in ratings.items():
            if v > max_val:
                max_key = k
        print(dishes[max_key][1])
        return
    elif len(dish_unrated_list) == 1:
        # only one dish unrated
        print("\nSuggested dish:")
        print(dishes[dish_list[0]][1])
        return
    else:
        ratings = dict()
        for item in dish_unrated_list:
            rating, status = user_rating(item, user_id, user_ratings)
            ratings[item] = rating
        max_ = max(ratings.iteritems(), key=operator.itemgetter(1))[0]
        print("\nSuggested dish:")
        print(dishes[max_][1])
        return





if __name__ == '__main__':

    with open('dishes.csv', 'rb') as f:
        reader = csv.reader(f)
        dishes = list(reader)

    user_ratings = json.load(open('./user_ratings.json'))

    dish_id = input("Enter dish ID: ")
    user_id = raw_input("Enter user ID: ")
    rating, status = user_rating(dish_id, user_id, user_ratings)
    print("\nRating: {:.5f} ({})\n".format(rating, status))

    user_id = raw_input("Enter user ID: ")
    num = input("Enter number of ingredients: ")
    ingredients_list = list()
    for i in range(1,num+1):
        temp = raw_input("Enter I{}: ".format(i))
        ingredients_list.append(temp)
    suggestion = find_dish(user_id, ingredients_list, dishes)
