# coding:utf-8
import os


def root(path):
    rootname = os.path.dirname(os.path.abspath(__file__))
    fullpath = os.path.join(rootname, path)
    return fullpath
