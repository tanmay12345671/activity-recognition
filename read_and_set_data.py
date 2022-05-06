# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:22:59 2022

@author: Administrator
"""

#Activity Recognition Project using Logistic Regression

'''Importing Necessary Libraries'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import logging as lg
os.chdir("E:\Ivy-Professional-School\Python\ML\Activity Recognition System\Code")
import log_info
get_log=log_info.getLog()


#reading the folder
os.chdir("E:\Ivy-Professional-School\Python\ML\Activity Recognition System\Dataset")
#entering im
get_log.info("Creating a list of directories from where we read our data")
usable_directory=[]

for i in range(len(os.listdir())):
    try:
        if os.listdir()[i][-3:]!="pdf":
            usable_directory.append(os.listdir()[i])
            get_log.info(os.listdir()[i]+" folder has added in the directory list")
    except Exception as e:
        get_log.info(e+" error has occured during add directory name in a list")               

usable_directory

#entering each of the folder and read all the dataset
for folders in usable_directory:
    try:
        get_log.info("Looping through each of the folder")
        os.chdir(f"E:\Ivy-Professional-School\Python\ML\Activity Recognition System\Dataset\{folders}")
        get_log.info("Entering in the folder named as "+folders)
    except Exception as e:
        get_log.info("Error occured during changing the directory the error is "+e)
    for files in os.listdir():
        get_log.info("Reading the file named as "+files)
        try:
            with open(f"E:\\Ivy-Professional-School\\Python\\ML\\Activity Recognition System\\Dataset\\{folders}\\{files}","r") as op:
                lines = op.readlines()
                added_name=files[:files.find(".")]+"_ch.csv"
                get_log.info("Create a copy cleaned version newly named "+added_name)
                with open(f"E:\\Ivy-Professional-School\\Python\\ML\\Activity Recognition System\\Dataset\\{folders}\\{added_name}","w") as op:
                    for line in lines[4:]:
                        op.write(line)
        except Exception as e:
            get_log.info("Error occured during entering in the files of folders, error is "+e)

        
















