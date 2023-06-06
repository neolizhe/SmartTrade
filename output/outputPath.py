# coding:utf-8
import os


def output(path):
    rootname = os.path.dirname(os.path.abspath(__file__))
    fullpath = os.path.join(rootname, path)
    return fullpath
