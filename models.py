import csv

import cv2


# def readCSV():
#     with open('./data/students.csv', 'rt')as f:
#         data = csv.reader(f)
#         return data
def readCSV():
    url = open("./data/student.csv", "r")
    read_file = csv.reader(url)
    return read_file
    url.close()


def writeCSV(arrList):
    url = open("./data/student.csv", "a+", newline='')
    write_file = csv.writer(url)
    write_file.writerow(arrList)
    url.close()
