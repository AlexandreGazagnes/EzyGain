#!/usr/bin/env python3
# -*- coding: utf-8 -*-


########################################################
########################################################
#       EzyGain
########################################################
########################################################


########################################################
#       Import
########################################################

# built in
import os, logging, sys, time, random, inspect, subprocess, pickle, datetime
# from math import ceil
# import itertools as it
from pprint import pprint
from collections import OrderedDict, Iterable, Counter

# data management
# from cloudant.client import CouchDB
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import Perceptron, RidgeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import LinearSVC, NuSVC

# from sklearn.model_selection import train_test_split

# from sklearn.dummy import DummyClassifier
# from sklearn.linear_model import LogisticRegression, LinearRegression

# from sklearn.metrics import accuracy_score
# from sklearn.utils import shuffle


########################################################
#       Logging and warnings
########################################################

# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)
l = logging.INFO
logging.basicConfig(level=l, format="%(levelname)s : %(message)s")
info = logging.info

# import warnings
# warnings.filterwarnings('ignore')


########################################################
#       Graph settings
########################################################

# %matplotlib
# sns.set()


########################################################
#       Filepaths
########################################################

FOLDER       = "/home/alex/Bureau/EzyGain/"
DATA_FOLDER  = FOLDER + "data/" 


########################################################
#       Consts
########################################################

# bdd
URL         = "https://a95ba22a-0143-4f5c-9c4f-be4a07240991-bluemix.cloudant.com/sample_data"
BDD_URL     = "https://themakeedentseentstionsp:3517b364138d149120c20951690d8f83591b7c16@a95ba22a-0143-4f5c-9c4f-be4a07240991-bluemix.cloudant.com/sample_data/"
USERNAME    = "themakeedentseentstionsp"
PASSWORD    = "3517b364138d149120c20951690d8f83591b7c16"
ALL         = "_all_docs"


########################################################
#       DataBase client
########################################################

# client = CouchDB(USERNAME, PASSWORD, url=URL, connect=True)


########################################################
#       DataFrame        
########################################################

def bdd_request(filename, stdout=False, save=True, meth="GET", ext="") :
    """
    desc    : exec a GET request on the bdd and save/return coresponding data
    
    pos arg : filename - str : the file you want to download
    
    opt arg : stdout - bool : if True, will return the stdout of os.system command, default False
              save   - bool : if True create a file else not, default True
              meth   - str  : API method in ["GET", "POST" "PUT"], default "GET"
              ext    - str  : the extension of file, ex ."json", default ""
    
    do      : save if needed the downloaded data
    
    raise   : AttributeError if args not valid
    
    return  :  the downloaded data in str format if stdout = True else None
    """
   
    # filename
    try    : filename = str(filename)
    except Exception as e : raise AttributeError(e)

    # stdout
    try    : stdout = bool(stdout)
    except Exception as e : raise AttributeError(e)

    # save 
    try    : save = bool(save)
    except Exception as e : raise AttributeError(e)

    # method
    try    : meth = str(meth)
    except Exception as e : raise AttributeError(e)
    meth = meth.upper()
    if meth not in ["GET", "POST" "PUT"] : 
        raise AttributeError('method ["GET", "POST" "PUT"]')
   
    # command
    url    = BDD_URL + filename
    cmd    = "curl -X {} {}".format(meth, url)
    info(cmd)

    # exec 
    response = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE)
    if response.returncode or response.stderr : return response.stderr

    # encode and cast
    txt = response.stdout.decode("utf-8")
    txt = txt.replace("false", "False").replace("true", "True").replace("null", "None")

    # save and stdout  
    if save   : 
        with open(str(DATA_FOLDER+filename+ext), "w") as f : f.write(txt)
    if stdout : return txt

    return None


####

# r = bdd_request(ALL, stdout=False, save=True)
# print(r)


##################################################################

def handle_all_docs(filename="_all_docs", save=True) : 
    """
    desc    : perform a bdd_request for '_all_docs', and create a clean df with 
              key/value
    
    pos arg : filename - str : the file you want, default "_all_docs" 
    
    opt arg : save - bool : if True the file will be saved, else just stream it, 
              default True
    
    do      : save the data if save = True  
    
    raise   : AttributeError if needed
    
    return  : a clean pd.DataFrame with columns [key, values]
    """

    # filename
    try                   : filename = str(filename) 
    except Exception as e : raise AttributeError(e)

    # save
    try                   : save = bool(save) 
    except Exception as e : raise AttributeError(e)

    # create df
    try    :   df = pd.read_json(DATA_FOLDER + filename)
    except :   df = pd.read_json(bdd_request(filename, save=save, stdout=True))
    df = df["rows"].values
    df = pd.DataFrame([pd.Series(i) for i in df])

    # clean 
    df["value"] = df.value.apply(lambda i : i["rev"])
    if (df.id.values == df.key.values).all() : df = df.drop("id", axis=1)

    # info
    info(df.columns)
    info(df.shape)

    return df

####

all_docs = handle_all_docs()

# here we have all keys
all_keys = all_docs.key.values


##################################################################

def download_all() : 
    """
    desc    : download and save each files (from key) in _all_docs
    
    pos arg : -
    
    opt arg : -
    
    do      : save files in DATA_FOLDER
    
    raise   : -
    
    return  : errors log list
    """

    # download index
    r = bdd_request(ALL, stdout=False, save=True)
    
    # grab ids
    all_docs = handle_all_docs()
    all_keys = all_docs.key.values

    # download
    errors = list()
    for k in all_keys :    
        try :                                        
            bdd_request(k, stdout=False, save=True)                    
        except Exception as e :                                        
            info(k)                                                    
            info(e)                 
            errors.append((k,e))

    # info nb of files in DATA_FOLDER
    cmd = "ls {}".format(DATA_FOLDER)
    l = len(list(os.popen(cmd)))
    info(l)

    return errors

####

# do not perform this download unless you are working on a local repo

# errors = download_all()
# print(errors)


##################################################################

def load_file(key, save=True) : 
    """
    desc    : load a file from his key, if possible read the corresponding file 
              else, download it, then try to cast the data as a pd.dataframe 
              or pd.Series object an return it 
    
    pos arg : key - str, the key of the file 
    
    opt arg : save - bool, if True download the file else just stream it 
    
    do      : - 
    
    raise   : AttributeError if needed
    
    return  : pd.DataFrame or pd.Series object 
    """

    # keys
    try                   : key = str(key) 
    except Exception as e : raise AttributeError(e)

    # save
    try                   : save = bool(save) 
    except Exception as e : raise AttributeError(e)

    # if file exists -> read it 
    if os.path.isfile(DATA_FOLDER + key)  : 
        try      : 
            df = pd.read_json(DATA_FOLDER + key)
        except   : 
            txt = open(str(DATA_FOLDER+key), "r").read()
            try    : df = eval(txt)
            except : df = txt 
            df = pd.Series(df)

    # if  not download it 
    else :
        txt = bdd_request(key, save=save, stdout=True)
        try      : 
            df = pd.read_json(txt)
        except   : 
            try    : df = eval(txt)
            except : df = txt
            df = pd.Series(df)

    return df


####


all_keys[0]
df = load_file(all_keys[0], save=False)


#####################################################################

def build_database(keys, save=False, threshold=0, drop_na=True) : 
    """
    desc    : load a files from a list of keys. build a  - database - with key, 
              _data (pd.DataFrame or pd.Series) and various info regardin this _data.  

    
    pos arg : keys - Iterable(list, set, pd.series), the keys of the files you want 
    
    opt arg : save - bool, if True download the file else just stream it
              threshold - int, the number of files you want to load, if 0 : all, 
              default 0
              drop_na - bool, if True drop Null _data 
    
    do      : save if needed
    
    raise   : various AttributeError coresponding args
    
    return  : pd.DataFrame object, with key as index and cols 
    """
    
    # keys 
    try                   : keys = list(keys) 
    except Exception as e : raise AttributeError(e)

    # save
    try                   : save = bool(save) 
    except Exception as e : raise AttributeError(e)

    # threshold
    try                   : threshold = int(threshold) 
    except Exception as e : raise AttributeError(e)

    # drop_na
    try                   : drop_na = bool(drop_na) 
    except Exception as e : raise AttributeError(e)

    # threshold if needed
    if threshold>0 : _keys = keys[:threshold]
    else           : _keys = keys
    
    # create list of df
    df_list = [(key, load_file(key, save=save)) for key in _keys]

    # add type and shapes
    _df_list = [  ( (i, type(j), j.shape, j) if  ( isinstance(j, pd.Series) 
                                  or isinstance(j, pd.DataFrame) ) 
                                  else ( i, type(j), (-1, -1),j)) 
                                  for i, j in  df_list ]

    # handle pd.Series 
    _df_list = [  ( (i, j, k, l) if len(k) ==2 else (i, j, (k[0],-1), l)) 
                for i,j, k, l in _df_list]

    # just keep nb of feat
    _df_list = [  (i, j, k[0], k[1], l) for i, j, k,l in _df_list]

    # init database
    db = pd.DataFrame(_df_list, columns=["key", "_type", "_len", "_feat", "_data"])

    # reindex
    db = db.set_index("key")

    # drop_na
    if drop_na : 
        idxs = db.loc[db["_len"] == 0, :].index
        db = db.drop(idxs, axis=0)

    # convert DataFrame of len(1) in pd.Series      
    new_data = [ ( pd.Series(db.loc[i, "_data"].iloc[0, :]) 
                        if (     (db.loc[i, "_len"] == 1) 
                              and (isinstance(db.loc[i, "_data"], pd.DataFrame))) 
                    else db.loc[i, "_data"])
                    for i in db.index]


    db["_data"] = new_data

    # updade _len, _feat, _type
    new_len  = [len(db.loc[i, "_data"]) for i in db.index ]

    new_feat = [    (      len(db.loc[i, "_data"].columns) 
                      if   isinstance(db.loc[i, "_data"], pd.DataFrame) 
                      else -1 )
                      for  i in db.index]

    new_type = [type(db.loc[i, "_data"]) for i in db.index ]

    db["_type"] = new_type
    db["_len"]  = new_len
    db["_feat"] = new_feat

    return db

####

# if reload try to del previous db to free space on memory
try     : del db
except  : pass
try     : del _db
except  : pass
try     : del db_meta 
except  : pass

db = build_database(all_keys, save=True, threshold=0)

##################################################################

# save and load if needed
def save_db(db, filename="db.pk") :

    path = FOLDER+filename
    with open(path, 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(db)
    return 0


def load_db(filename="db.pk") : 

    path = FOLDER+filename
    with open(path, 'rb') as fichier:
        depickler = pickle.Unpickler(fichier)
        return depickler.load()



####

# save_db(db)
# db = load_db()


####################################################################

def manage_meta(db, force_up_level=True, main_cat=True) : 
    """
    desc    : from a dabase, identify "meta_params" ie features in _data with 
              unique == 1, create a "meta_params" to store them and drop this 
              feature in _data for both series and dataframe
              if needed force a sub set of these features to be stored not 
              in meta_params but in db.columns as a global feature

    pos arg : db - pd.DataFrame, the database 
    
    opt arg : force_up_level - bool, if True delete some meta_params keys and 
              store them in db.columns as a global feature, if not let them 
              stored in meta_params default True
              main_cat - bool, if True add group table and feature in one feature 
              named 'main_cat', then drop them, default True
   
    do      : - 
    
    raise   : AttributeError if needed 
    
    return  : pd.DataFrame with meta_params feature
    """

    # db
    if not isinstance(db, pd.DataFrame) : 
        raise AttributeError("invalid type for db")
    _db = db.copy()

    # force_up_level
    try                   : force_up_level = bool(force_up_level) 
    except Exception as e : raise AttributeError(e)

    # main_cat
    try                   : main_cat = bool(main_cat) 
    except Exception as e : raise AttributeError(e)

    metas = list()
    objs  = list()
    for i in _db.index : 

        # init our dataframe / series  
        obj = _db.loc[i, "_data"].copy()
        # info(obj)

        # if empty obj : next iter
        if len(obj) < 1 : 
            metas.append(dict())
            objs.append(obj)
            continue

        # if pd.DataFrame
        if isinstance(obj, pd.DataFrame ) : 
        
            #indetify meta candidates  
            meta_list =  list()
            for feat in obj.columns : 
                try : 
                    if len(obj[feat].unique()) == 1 : 
                        meta_list.append(feat)
                except : 
                    pass

            # create our dict
            meta_dict = {feat: obj.iloc[0][feat] for feat in meta_list}

            # del meta
            # info(type(obj))
            obj = obj.drop(meta_list, axis=1) 

            # record our(s) feature(s) in meta
            if len(obj.columns) >= 1   : meta_dict["feature"] = list(obj.columns)
            else                       : meta_dict["feature"] = list()

            # update meta and obj
            metas.append(meta_dict)
            info(obj)
            objs.append(obj)

        # if pd.Series
        else : 

            # same thing than for pd.DataFrame objects
            meta_list = [ i for i in obj.index if (
                                            (not isinstance(obj[i], list))
                                        and (not isinstance(obj[i], dict)) ) ] 

            meta_dict = { i: obj[i] for i in meta_list }
            obj = obj.drop(meta_list) 
            meta_dict["feature"] = list(obj.index)

            # update meta and obj
            metas.append(meta_dict)
            info(obj)
            objs.append(obj)

    # check good shapes
    if len(metas) != len(_db) : raise ValueError("pb len de metas")
    if len(objs)  != len(_db) : raise ValueError("pb len de objs")

    # # info if needed
    # info(metas)
    # info(objs)

    # update meta and _objs
    _db["meta_params"] = metas
    _db["_data"] = objs

    # force meta_params to be in db.columns 
    if force_up_level : 

        # force new_meta struct, and create _db feat from meta if needed 
        new_meta        = list()
        _id_list        = list()
        _rev_list_      = list()
        feature_list    = list()
        updatedAt_list  = list()
        table_list      = list()
        createdAt_list  = list()

        for idx in _db.index : 
            d = _db.loc[idx, "meta_params"]
            k_list = d.keys()
            for k in ["_id", "_rev", "feature", "updatedAt", "table", "createdAt"] : 
                if not k in k_list     : d[k] = None
            # if not "feature" in k_list : d[k] ="None"

            new_meta.append(d)
            _id_list.append(d["_id"])
            _rev_list_.append(d["_rev"])
            feature_list .append(d["feature"])
            updatedAt_list.append(d["updatedAt"])
            table_list.append(d["table"])
            createdAt_list.append(d["createdAt"])

        _db["meta_params"] = new_meta
        _db["_id"]         = _id_list       
        _db["_rev"]        = _rev_list_     
        _db["feature"]     = feature_list  
        _db["updatedAt"]   = updatedAt_list 
        _db["table"]       = table_list     
        _db["createdAt"]   = createdAt_list 

        # delete redondodant info between _db.columns and _db.meta_params
        new_meta        = list()
        for idx in _db.index : 
            d = _db.loc[idx, "meta_params"]
            for k in ["_id", "_rev", "feature", "updatedAt", "table", "createdAt"] : 
                try    : d.pop(k)
                except : pass
            new_meta.append(d)

        _db["meta_params"] = new_meta
        _db["feature"] = _db.feature.apply(lambda i : ("".join(i) if i else None))

    # main cat
    if main_cat and force_up_level: 
        _db["main_cat"] = ["{}_{}".format(i, j) for i, j in zip(_db.table, _db.feature)]
        _db = _db.drop(["table", "feature"], axis=1)

    return _db

####

db_meta = manage_meta(db)
cats = db_meta.main_cat.unique()
print(cats)

# if reload try to del previous db to free space on memory
try    : del db
except : pass
try    : del _db
except : pass


##################################################################

def explore_main_cats(db) : 
    """
    """

    cats = db_meta.main_cat.unique()

    for c in cats : 
        pprint(c)
        try : 
            pprint(db_meta.loc[db_meta.main_cat == c, "_data"].iloc[0].iloc[0])
        except : 
            pprint("error")
        print("\n\n\n")

####

explore_main_cats(db_meta)


##################################################################

def group_by_feat(db, feat="main_cat", verbose=True) : 
    """
    """

    # db
    if not isinstance(db, pd.DataFrame) : 
        raise AttributeError("invalid type for db")

    # feat
    try                   : feat = str(feat) 
    except Exception as e : raise AttributeError(e)

    if feat not in db.columns : 
        raise AttributeError("feat not in db.columns")


    # create a spec df
    df              = pd.DataFrame({"key":db.index.values, feat:db[feat].values}) 

    # group and handle
    grouped         = df.groupby(feat)
    grouped         = {idx : list(_df.key.values)for idx, _df in grouped}
    
    # info value_counts() like method
    if verbose : 
        grouped_count   = {idx : len(elem) for idx, elem in grouped.items()}
        pprint(grouped_count)
        pd.Series(grouped_count).plot(kind="bar")
        plt.xlabel("file_type")
        plt.ylabel("count")

    return grouped

####

grouped  = group_by_feat(db_meta)


##################################################################

def focus_on_activities(db_meta) : 
    """
    """

    # copy data
    _db_meta = db_meta.copy()

    # extract only "activites" and create a meta_feature
    activities = _db_meta.loc[_db_meta.main_cat == "activities_gameData", :]
    activities["_type"] = [i.meta_params["activityName"] for _, i in activities.iterrows()]

    # uniquye type
    activities_type_unique = set([i.meta_params["activityName"] for _, i in activities.iterrows()])

    # split activities in main, and sub
    def split_activity_type(act) : 
        if not isinstance(act, str)   : raise AttributeError("string expected")
        if "static_analysis"   in act : return "static_analysis", act.replace("static_analysis_", "")
        if "walking_landscape" in act : return "walking", act.replace("walking_", "")
        else                          : return act, None

    # update activity feature and drop old one
    activities["_type"]     = activities._type.apply(split_activity_type)
    activities["main_type"] = activities._type.apply(lambda i : i[0])
    activities["sub_type"]  = activities._type.apply(lambda i : i[1])
    activities              = activities.drop("_type", axis=1)

    return activities


####

activities = focus_on_activities(db_meta)

try    : del db_meta
except : pass
try    : del db
except : pass
try    : del _db
except : pass


##################################################################

def explore_activites(activities) : 
    """
    """
    
    act_count = activities.main_type.value_counts()
    act_count.plot(kind="bar")
    plt.title("distribution of activites main type")
    plt.xlabel("activites"); plt.ylabel("count")
    plt.show()

    act_count = act_count.drop("walking")
    act_count.plot(kind="bar")
    plt.title("distribution of activites main type")
    plt.xlabel("activites"); plt.ylabel("count")
    plt.show()


explore_activites(activities)

##################################################################

def flatten_walking(activities) : 
    """
    """

    # select only walking activites
    main_activities_unique = activities.main_type.unique()
    walking = activities.loc[activities.main_type=="walking", :].copy()

    # flatten (normalize) data
    walking_datas = list()
    walking_meta_params = list()

    for i_num, i_val in enumerate(walking.index): 

        walk                =  walking.loc[i_val, :]

        walk_meta_params    = walk.meta_params
        # info(walk_meta_params)

        walk_data           = walk["_data"]["gameData"]
        # info(walk_data)
        # print("\n"*3)

        # sainty check
        if not isinstance(walk_data, dict) : 
            if len(walk_data) == 1 : 
                walk_data = walk_data[0]
                if not  isinstance(walk_data, dict) : raise TypeError("expected dict")
            else : raise TypeError("expected len _data = 1 ")

        # get nb of chuncks
        # BE CAREFULL SOME WALKS HAS NO CHUNKS --> ERRORS
        if "gaitDataChunks" in walk_data.keys() : 
            walk_data["nb_chunks"] = len(walk_data["gaitDataChunks"])
        else : 
            walk_data["nb_chunks"] = 0

        # force iterable  
        def force_iter(elem) : 
            if     isinstance(elem, str)      : return [elem,]
            if not isinstance(elem, Iterable) : return [elem,]
            else                              : return elem
        walk_data = {i : force_iter(elem) for i, elem in walk_data.items() }    

        
        if walk_data["nb_chunks"][0] :                  
            # try to flatten gaitDataChunks
            gaitDataChunks = pd.DataFrame(walk_data["gaitDataChunks"])

            for c, elem in gaitDataChunks.iteritems() : 
                walk_data[c] = list(elem.values)

            del walk_data["gaitDataChunks"]

            # try to flatten metronomValues

            # walk_data["metronomValues_avgMetronomSpeed"] = walk_data["metronomValues"]["avgMetronomSpeed"] 
            # del walk_data["metronomValues"]["avgMetronomSpeed"]
            # walk_data["metronomValues_chunks"] = walk_data["metronomValues"]["chunks"] 

            ######################################
            try : 
                del walk_data["metronomValues"]
            except : 
                pass
            ##########################################

        # sanity check
        for k, v in walk_data.items() : 
            if not isinstance(v, Iterable) : raise TypeError("{} is not iterable".format(k))

        # delete None values
        walk_data = {k: v for k, v in walk_data.items() if len(v)>0}

        # pop _data in meta_params
        del_item = list()
        for k, v in walk_data.items() : 
            if len(v) == 1 : 
                walk_meta_params[k] = v[0]
                del_item.append(k)
            else :
                walk_meta_params[k] = "Iterable"

        walk_data = {k: v for k, v in walk_data.items() if k not in del_item}

        walking_datas.append(walk_data)
        walking_meta_params.append(walk_meta_params)

    # update _data and _meta
    walking["_data"]        = walking_datas
    walking["meta_params"]  = walking_meta_params 

    # pop meta in upper level
    meta_params  = pd.DataFrame([pd.Series(i) for i in walking["meta_params"]], index=walking["meta_params"].index)
    walking = walking.join(meta_params, how="right", rsuffix="_meta")
    walking = walking.drop("meta_params", axis=1)


    return walking



####

walking = flatten_walking(activities)

##################################################################

def explore_walking(walking) : 

    print(walking.columns)

    numeric_feats = ["distance", "duration", "nb_chunks", "patientWeight", "steps", "speed"]
    fig, axs = plt.subplots(1, len(numeric_feats), figsize=(12, 5))
    for i, feat in enumerate(numeric_feats) : 
        feats = walking[feat].replace("Iterable", np.nan).dropna().astype(np.float32)
        axs[i].hist(feats.dropna())
        axs[i].set_ylabel("count")
        axs[i].set_xlabel(feat)

    plt.suptitle("Numerical features distribution") 
    plt.show()

    cat_feats = ["deviceSerialNumber", "sub_type"]
    fig, axs = plt.subplots(1, len(cat_feats), figsize=(12, 5))
    for i, feat in enumerate(cat_feats) : 
        feats = walking[feat].replace("Iterable", np.nan).dropna().value_counts()
        feats.plot(kind="bar", ax=axs[i])
        axs[i].set_ylabel("count")
        axs[i].set_xlabel(feat)
    plt.suptitle("Categorical features distribution") 
    plt.show()

    df = walking.loc[:, ["PatientId", "steps"]].replace("Iterable", np.nan).dropna(how="any", axis=0)
    grouped = df.groupby("PatientId")
    grouped = pd.Series({k: len(val) for k, val in grouped })
    grouped.index = ["patient "+str(i) for i in range(len(grouped))]
    grouped.plot(kind="bar")
    plt.xlabel("patient - anonymous")
    plt.ylabel("count")
    plt.title("number of seance by patient")
    plt.show()


    # time
    original_dates = walking.createdAt.sort_values(ascending=False).values.copy()
    original_dates[:10]

    # warning, time in MiliSeconds
    second_dates = [int(i/1000) for i in original_dates]

    def _handle_timestamp(t) : 
        t = datetime.datetime.fromtimestamp(t)
        txt = "{}-{}-{} {}:{}:{}".format(   t.day, t.month, t.year,
                                            t.hour, t.minute, t.second)
        return txt

    # pprint([(i, _handle_timestamp(i)) for i in second_dates][:3])
    # print()
    # pprint([(i, _handle_timestamp(i)) for i in second_dates][-3:])

    one_day = 24 * 60 * 60
    day_dates = [int(i /(one_day)) for i in second_dates]
    nb_of_days = day_dates[0] - day_dates[-1]
    nb_of_weeks = int(nb_of_days / 5)

    ys, xs, _ = plt.hist(pd.Series(day_dates), bins=nb_of_days)
    plt.xlabel("day")
    new_xs = [i for i in xs if not i%5]
    new_labels = list(range(len(new_xs))) 
    plt.xticks(new_xs, new_labels)
    plt.ylabel("count")
    plt.title("frequency by time (delta = weeks)")
    plt.show()

    ys, xs, _ = plt.hist(day_dates, bins=nb_of_weeks)
    plt.xlabel("day")
    new_labels = list(range(len(xs))) 
    plt.xticks(xs, new_labels)
    plt.ylabel("count")
    plt.title("frequency by time (delta = week)")
    plt.show()

####

explore_walking(walking)


##################################################################


##################################################################
#       Garbage
##################################################################


def _handle_timestamp(t) : 
    t = datetime.datetime.fromtimestamp(t / 1e3)
    txt = "{}-{}-{} {}:{}:{}".format(   t.day, t.month, t.year,
                                        t.hour, t.minute, t.second)
    return txt


def find_candidate_feat(database) : 
    """find all feat (pd.DataFrame) or index (pd.Series) of the "_data" in the database"""
    candidate_feat = list()
    for idx in database.index : 
        obj = database.loc[idx, "_data"]
        cols = obj.columns if isinstance(obj, pd.DataFrame) else obj.index
        candidate_feat += list(cols)
    candidate_feat = list(set(candidate_feat))
    return candidate_feat


def find_valid_feat(database, candidate_feat) : 
    """find each of candidate_feat are in ALL the "_data" feats in the database"""
    valid_feat = list()
    for candidate in candidate_feat : 
        in_list = list()
        for idx in database.index : 
            obj = database.loc[idx, "_data"]
            is_in = ( (candidate in obj.columns) if isinstance(obj, pd.DataFrame) 
                                                 else (candidate in obj.index) )
            in_list.append(is_in)

        if pd.Series(in_list).all() : valid_feat.append(candidate)
    return valid_feat







def flatten_data(db) : 
    """
    desc    : try to flatten the _data obj from db : look for features in _data
              if one this feature is an Iterable (dict for inst) try to transform
              this feature in a up level dataframe feature
        
    pos arg : db - pd.DataFrame, the database 
    
    opt arg : - 
    
    do      : - 
    
    raise   : AttributeError if needed
    
    return  : pd.DataFrame object, with key as index and cols 
    """

    # db
    if not isinstance(db, pd.DataFrame) : 
        raise AttributeError("invalid type for db : not a pd.DataFrame obj")
    if not "meta_params" in db.columns : 
        raise AttributeError("invalid shape for db : please call manage_meta()")
    _db = db.copy()

    metas = list()
    objs  = list()
    for i in _db.index : 

        # init our dataframe / series
        meta_dict   = _db.loc[i, "meta_params"].copy()
        obj         = _db.loc[i, "_data"].copy()
        # info(meta_dict)
        # info(obj)
    
        try : 

        # if pd.DataFrame
        if isinstance(obj, pd.DataFrame ) : 

            # if empty obj : next iter
            if len(obj) < 1 : 
                metas.append(meta_dict)
                objs.append(obj)
                continue

            if len(obj.columns) >= 1 :
                pass
            else : 
                metas.append(meta_dict)
                objs.append(obj)
                continue

            # and grab low level feat to front cols features
            sub_feature=list()
            for feat in obj.columns : 
                sample = obj.iloc[0][feat]
                if not isinstance(sample, dict) : 
                    continue 

                sample_keys = sorted(list(sample.keys()))
                for f in sample_keys : 
                    obj[f] = obj[feat].apply(lambda d : d[f])
                    sub_feature.append(f)
                obj = obj.drop(feat, axis=1)

            meta_dict["sub_feature"] = list(sub_feature)
            metas.append(meta_dict)
            objs.append(obj)

        else : 

            #################

            # PLEASE CODE THIS

            ##################

            metas.append(meta_dict)
            objs.append(obj)

    # check valid shapes
    if len(metas) != len(_db) : raise ValueError("pb len de metas")
    if len(objs) != len(_db)  : raise ValueError("pb len de objs")

    # info if  needed 
    info(metas)
    info(objs)

    # _db
    _db["meta_params"] = metas
    _db["_data"] = objs

####

db_meta_flatten = flatten_data(db_meta)
