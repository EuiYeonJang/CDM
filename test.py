import pickle as pkl
import json

# initializing dictionary
# test_dict = {'Gfg': 1, 'is': 2, 'best': 3}
#
# # printing original dictionary
# print("The original dictionary is : " + str(test_dict))

# # using encode() + dumps() to convert to bytes
# res_bytes = json.dumps(test_dict).encode('utf-8')
#
# # printing type and binary dict
# print("The type after conversion to bytes is : " + str(type(res_bytes)))
# print("The value after conversion to bytes is : " + str(res_bytes))
#
# # # using decode() + loads() to convert to dictionary
# res_dict = json.loads(res_bytes.decode('utf-8'))
# #
# # # printing type and dict
# print("The type after conversion to dict is : " + str(type(res_dict)))
# print("The value after conversion to dict is : " + str(res_dict))

with open("liwc.dict", "rb") as f:
    s = f.read()
    print("The type after conversion to bytes is : " + str(type(s)))
    print("The value after conversion to bytes is : " + str(s))
    print(len(s))

    x = s.decode()
    print(x)
    print(type(x))
    print(len(x))

amont_brak = 0
# l = x.split('"') # the [1::2] is a slicing which extracts odd values
# print(l)

category_list = []

category_dict_low = {}
category_dict_high = {}
for i in range(len(x)):
    # print(x[i])


    #NEW CATEGORY IF 'asS'
    if x[i:i+2] == 'as':
        if x[i:i+4] == 'assS':   #HIGH LEVEL CATEGORY
            for j in range(0,100):
                if x[i+j+5] == '\'':
                    break
            category_list.append(x[i+5:i+j+5])
            category_dict_low[x[i + 5:i + j + 5]] = []
            current_cat = x[i + 5:i + j + 5]
        elif x[i:i+3] == 'asS':  #SUB CATEGORY
            for j in range(0,100):
                if x[i+j+4] == '\'':
                    break
            category_list.append(x[i+4:i+j+4])
            category_dict_low[x[i+4:i+j+4]]=[]
            current_cat = x[i+4:i+j+4]
    elif x[i:i+2] == 'aS':
        for j in range(0, 100):
            if x[i + j + 3] == '\'':
                break
        print('word',x[i + 3:i + j + 3])
        category_dict_low[current_cat].append(x[i + 3:i + j + 3])
        print(current_cat, category_dict_low[current_cat])
    # elif x[i:i+2] == 'S\'':
    #     print('ok ')
    #     for j in range(0, 100):
    #         if x[i + j + 3] == '\'':
    #             break
    #     category_dict_low[x[i + 5:i + j + 5]] = []
    #     current_cat = x[i + 5:i + j + 5]
print(category_list)
print(len(category_list))
# print(category_dict_low)
# print(category_dict_low['PRONOUN'])
    # z = list(s)
    # print(z)

# for i in range(len(x)):
#     prin
    # if str(s[i]) == 'as':
    #     print('yes')
    # else:
    #     print('no')
    # print(i, str(s[i]))
    # print(bytes.fromhex(str(s[i])))

# print(s)

    # res_s = json.loads(s.decode('utf-8'))
    # # print()
    # print("The type after conversion to bytes is : " + str(type(res_s)))

    # print("The value after conversion to bytes is : " + str(res_bytes))

    # print(s)
    # print(type(s))
    # res_dict = json.loads(s.decode('utf-8'))
    # adj_list = pkl.load(f)
    #
    # s = f.read()
    # self.whip = ast.literal_eval(s)