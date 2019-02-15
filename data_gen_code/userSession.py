#!/usr/bin/python
# -*- coding:utf-8 -*-
from pyspark.sql import SparkSession
from  pyspark.sql.types import StringType
import pyspark.sql.functions as func
from datetime import datetime, timedelta
from pyspark.sql import Row
import sys
import math
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')

##### 函数注册
def parse_session_test(items,bhv_time,target_item_id):
    bhv_list = items.split(',')
    if len(bhv_list)<5 or len(bhv_list) > 100:
        return ''
    sortByTime = {}
    for item in bhv_list:
        time = int(item.split(':')[1])
        item_id = item.split(':')[0]
        if time < bhv_time:
            sortByTime[time] = item_id
    # sort sortByTime
    rd_list = []
    for time in sorted(sortByTime.keys()):
        item_id = sortByTime[time]
        if len(rd_list)>=20:
            break
        rd_list.append(item_id)
    return '#'.join(rd_list)
# item_session = func.udf(parse_session,StringType())
item_session_test = func.udf(parse_session_test,StringType())


def parse_session(items):
    bhv_list = items.split(',')
    sortByTime = {}
    for item in bhv_list:
        time = int(item.split(':')[1])
        item_id = item.split(':')[0]
        sortByTime[time] = item_id
    # sort sortByTime
    rd_list = []
    for time in sorted(sortByTime.keys()):
        item_id = sortByTime[time]
        rd_list.append(item_id)
    return '#'.join(rd_list)
item_session = func.udf(parse_session,StringType())




def cat_session(cat_dict):
    def parse_cat_session(col):
        cat1 = []
        cat2 = []
        cat3 = []
        cat4 = []
        bhv_list = col.split('#')
        for item in bhv_list:
            try:
                item = int(item)
            except:
                continue
            if item in cat_dict:
                cat1.append(str(cat_dict[item][0]))
                cat2.append(str(cat_dict[item][1]))
                cat3.append(str(cat_dict[item][2]))
                cat4.append(str(cat_dict[item][3]))
        return '#'.join(cat1)+';'+'#'.join(cat2)+';'+'#'.join(cat3)+';'+'#'.join(cat4)
    return func.udf(parse_cat_session,StringType())


# def add_neg_sample(p,totalItems):
#     click_items_list = p['item_hist'].split('#')
#     click_item_normal = []
#     click_items_set = set()
#     for item in click_items_list:
#         if item.isdigit():
#             click_items_set.add(int(item))
#             if len(click_item_normal)<20:
#                 click_item_normal.append(item)
#     neg_items = totalItems-click_items_set
#     item_hist_str = '#'.join(click_item_normal)
#     rd_list = []
#     rd_list.append(Row(user_id=p.user_id,item_id=p.item_id,label=p.label,item_hist=item_hist_str))
#     neg_sample = np.random.choice(list(neg_items),10,False)
#     for item in neg_sample:
#         rd_list.append(Row(user_id=p.user_id,item_id=int(item),label=0,item_hist=item_hist_str))
#     return rd_list


def add_neg_sample(p,totalItems):
    click_items_list = p['item_hist'].split('#')
    click_items_set = set()
    if len(click_items_list)<5 or len(click_items_list) > 100:
        return []
    for item in click_items_list:
        if item.isdigit():
            click_items_set.add(int(item))
    neg_items = totalItems-click_items_set
    neg_sample = np.random.choice(list(neg_items),5,False)
    rd_list = []
    for i in range(len(click_items_list)-1,max(len(click_items_list)-20,4),-1):
        item_id = click_items_list[i]
        item_hist = click_items_list[max(i-20,0):i]
        item_hist_str = '#'.join(item_hist)
        rd_list.append(Row(user_id=p.user_id,item_id=item_id,label=1,item_hist=item_hist_str))
        for item in neg_sample:
            rd_list.append(Row(user_id=p.user_id,item_id=int(item),label=0,item_hist=item_hist_str))
    return rd_list

if __name__ == "__main__":

    # ############### spark 环境 ##########################
    spark = SparkSession \
        .builder \
        .appName("train_test_data") \
        .enableHiveSupport() \
        .getOrCreate()
    
    ### 获取类目信息 #######
    cat_map = spark.sql(" select item_id ,phy_category1_id,phy_category2_id,phy_category3_id,phy_category4_id from dw.dwd_goods_item_phy_category_i where ds = '2019-02-11'")
    cat_collect = cat_map.collect()
    dict_cat = {}
    for row in cat_collect:
        dict_cat[row[0]] = row[1:]
    
    totalItems = dict_cat.keys()


    #### step 1 获取10天的训练样本数据 #####
    # print "*"*40+"获取10天的训练样本数据"+"*"*40

    # bhv_data = spark.sql("select user_id,act_obj as item_id,bhv_type,bhv_time,ds from data_mining.rec_user_behavior where scn = 'home' and tpl = 'guess_like' and bhv_type in ('show','click') and ds >= '2018-07-15' and ds < '2018-08-01'")

    # #### step 2 构造正样本 #####
    # print "*"*40+"构造正样本"+"*"*40
    # pos_label = bhv_data.where("bhv_type = 'click'").withColumn('label',func.lit(1)).select('user_id','item_id','label','bhv_time','ds')

    # bhv_data = spark.sql("SELECT user_id,item_id,max(label) as label,max(bhv_time) as bhv_time,ds \
    #                       FROM ( \
    #                         SELECT user_id,act_obj as item_id,if(bhv_type = 'click',1,0) AS label,bhv_time,ds \
    #                         FROM data_mining.rec_user_behavior where scn = 'home' and tpl = 'guess_like' and bhv_type in ('show','click') and ds >= '2018-07-15' and ds < '2018-08-01' ) a \
    #                       group by user_id ,item_id,label,ds")


    #### step 3 生成item session数据 #####

    ###
    # lastTenDay = datetime.today() + timedelta(-180)
    # today = datetime.today()
    # today_format = today.strftime('%Y-%m-%d')
    # ds = lastTenDay.strftime('%Y-%m-%d')

    userItemClickHist = spark.sql("select user_id ,concat_ws(',',collect_set(term)) as items from \
                        (   \
                            select user_id,concat(item_id,':',bhv_time) as term from  \
                            ( \
                                select user_id,act_obj as item_id,bhv_time from data_mining.rec_user_behavior where ds > '2018-07-15' and ds <'2018-08-01' and bhv_type = 'click' and obj_type = 'item' and act_obj is not null \
                            ) a \
                            join (select distinct id from dw.dim_item where is_invalid = 0 and phy_category1_id not in (1002269,1002860) ) b \
                            on item_id = id \
                        ) b \
                        group by user_id")

    print "*"*40+"生成user_item session"+"*"*40
    # userItemClickHistWithLabel   = userItemClickHist.join(bhv_data,'user_id')

    # userSession = userItemClickHist.select('user_id','item_id','label',item_session('items','bhv_time','item_id').alias('item_hist'))
    userSession = userItemClickHist.select('user_id',item_session('items').alias('item_hist'))

    # userSession.show(10,False)

    #### step 2 构造负样本 #####
    sample = userSession.rdd.flatMap(lambda p: add_neg_sample(p,set(totalItems))).toDF()
    sample.show(10,False)


    #### step 4 生成 cat_session 数据 ######

    print "*"*40+"生成user_cat session"+"*"*40



    sample = sample.withColumn('cat_session',cat_session(dict_cat)("item_hist"))

    split_col = func.split(sample['cat_session'], ';')

    sample = sample.withColumn('cat1_session', split_col.getItem(0)) \
                            .withColumn('cat2_session', split_col.getItem(1)) \
                            .withColumn('cat3_session', split_col.getItem(2)) \
                            .withColumn('cat4_session', split_col.getItem(3))

    # sample.show(10,False)
    

    #### step 3 生成last item click feature ######

    sample = sample.withColumn('click_last_i', func.regexp_extract('item_hist', '(\d+$)', 1)) \
                    .withColumn('click_last_cat1', func.regexp_extract('cat1_session', '(\d+$)', 1)) \
                    .withColumn('click_last_cat2', func.regexp_extract('cat2_session', '(\d+$)', 1)) \
                    .withColumn('click_last_cat3', func.regexp_extract('cat3_session', '(\d+$)', 1)) \
                    .withColumn('click_last_cat4', func.regexp_extract('cat4_session', '(\d+$)', 1))


    sample = sample.join(cat_map,'item_id')
    sample = sample.select('item_hist','cat1_session','cat2_session','cat3_session','cat4_session','click_last_i','click_last_cat1','click_last_cat2','click_last_cat3','click_last_cat4','label','user_id','item_id','cat1','cat2','cat3','cat4')

    sample.show(10,False)
    ##### save sample ######
    today_format = '20180731'
    sample.write.format('csv').mode('overwrite').save("/user/ai/yanxuan/recommend/ltr/ds="+today_format+"/train_sample")


    # ##### test sample ######
    
    # #### step 1 获取后1天的样本数据 #####
    print "*"*40+"获取后1天的测试样本数据"+"*"*40
    test_bhv_data = spark.sql("SELECT user_id,item_id,max(label) as label,ds \
                          FROM ( \
                            SELECT user_id,act_obj as item_id,if(bhv_type = 'click',1,0) AS label,ds \
                            FROM data_mining.rec_user_behavior where bhv_type in ('show','click') and ds = '2018-08-01' ) a \
                          group by user_id ,item_id,label,ds")


    # print "*"*40+"生成user_item session"+"*"*40
    userItemClickHistWithLabel_test   = userItemClickHist.join(test_bhv_data,'user_id')

    test_sample = userItemClickHistWithLabel_test.select('user_id','item_id','label',item_session_test('items',func.lit(1533081600000),'item_id').alias('item_hist')).filter("item_hist!=''")

    # test_sample.show(10,False)

    # #### step 4 生成 cat_session 数据 ######

    print "*"*40+"生成user_cat session"+"*"*40



    test_sample = test_sample.withColumn('cat_session',cat_session(dict_cat)("item_hist"))

    split_col = func.split(test_sample['cat_session'], ';')

    test_sample = test_sample.withColumn('cat1_session', split_col.getItem(0)) \
                            .withColumn('cat2_session', split_col.getItem(1)) \
                            .withColumn('cat3_session', split_col.getItem(2)) \
                            .withColumn('cat4_session', split_col.getItem(3))

    # test_sample.show(10,False)
    

    # #### step 3 生成last item click feature ######

    test_sample = test_sample.withColumn('click_last_i', func.regexp_extract('item_hist', '(\d+$)', 1)) \
                    .withColumn('click_last_cat1', func.regexp_extract('cat1_session', '(\d+$)', 1)) \
                    .withColumn('click_last_cat2', func.regexp_extract('cat2_session', '(\d+$)', 1)) \
                    .withColumn('click_last_cat3', func.regexp_extract('cat3_session', '(\d+$)', 1)) \
                    .withColumn('click_last_cat4', func.regexp_extract('cat4_session', '(\d+$)', 1))

    test_sample = test_sample.join(cat_map,'item_id')
    test_sample = test_sample.select('item_hist','cat1_session','cat2_session','cat3_session','cat4_session','click_last_i','click_last_cat1','click_last_cat2','click_last_cat3','click_last_cat4','label','user_id','item_id','cat1','cat2','cat3','cat4')
    test_sample.show(10,False)
    # ##### save sample ######
    today_format = '20180731'
    test_sample.write.format('csv').mode('overwrite').save("/user/ai/yanxuan/recommend/ltr/ds="+today_format+"/test_sample")