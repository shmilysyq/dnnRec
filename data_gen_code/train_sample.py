import numpy as np


def add_neg_sample(p,totalItems):
    click_items_list = p['item_hist'].split('#')
    click_items_set = set()
    if len(click_items_list)<5 or len(click_items_list) > 100:
        print(click_items_list)
        return []
    for item in click_items_list:
        if item.isdigit():
            click_items_set.add(int(item))
    neg_items = totalItems-click_items_set
    neg_sample = np.random.choice(list(neg_items),5,False)
    rd_list = []
    for i in range(len(click_items_list)-1,2,-1):
        print(i)
        item_id = click_items_list[i]
        item_hist = click_items_list[max(i-20,0):i]
        item_hist_str = '#'.join(item_hist)
        print(item_hist_str)
        # rd_list.append(Row(user_id=p.user_id,item_id=item_id,label=1,item_hist=item_hist_str))
        for item in neg_sample:
            print(item_hist_str)
            # rd_list.append(Row(user_id=p.user_id,item_id=int(item),label=0,item_hist=item_hist_str))
    return rd_list

p={'item_hist':'1#2#3#4#5'}
totalItems = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}
add_neg_sample(p,totalItems)

# import numpy as np
# import matplotlib.pyplot as plt
# name_list=[]
# num_list=[]
# with open(r'C:\Users\shengyaqi\Desktop\userBhvCnt',encoding='utf-8') as cnt_file:
#     for line in cnt_file:
#         arr = line.split('\t')
#         num = int(arr[1])
#         name = int(arr[0])
#         if num<5 or name <5:
#             continue 
#         name_list.append(arr[0])
#         num_list.append(int(arr[1]))
# # plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
# # plt.figure(figsize=(60,1))
# # plt.show()