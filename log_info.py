# -*- coding: utf-8 -*-
"""
Created on Wed May  4 09:25:31 2022

@author: Administrator
"""
import os
os.chdir("E:\Ivy-Professional-School\Python\ML\Activity Recognition System\Code")


import logging
# def log_file():
#     logger = lg.getLogger(__name__)
#     logger.setLevel(lg.INFO)
#     formatter = lg.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
#     file_handler = lg.FileHandler('recognition.log')
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
    
    
def getLog():
    # Creating custom logger
    logger = logging.getLogger(__name__)
    # reading contents from properties file
    logger.setLevel(logging.INFO)
    # Creating Formatters
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    # Creating Handlers
    file_handler = logging.FileHandler('recognition.log')
    # Adding Formatters to Handlers
    file_handler.setFormatter(formatter)
    # Adding Handlers to logger
    logger.addHandler(file_handler)
    return logger    
    
    
    
    
    