import csv
import collections

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)


##### Fs contain the rows
##### Ls are to check that the first three attributes are unique identifiers 
F_age25 = []
L1 = []
with open('GermanCredit_age25.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            F_age25.append(row)
            L1.append([row[0], row[1], row[2]])
            line_count += 1
    print(f'Processed {line_count} lines.')

F_age35 = []
L2 = []
with open('GermanCredit_age35.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            F_age35.append(row)
            L2.append([row[0], row[1], row[2]])
            line_count += 1
    print(f'Processed {line_count} lines.')

F_sex = []
L3 = []
with open('GermanCredit_sex.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            F_sex.append(row)
            L3.append([row[0], row[1], row[2]])
            line_count += 1
    print(f'Processed {line_count} lines.')

for i in range(len(L1)):
    if not compare(L1[i], L2[i]) or not compare(L1[i], L3[i]) :
        print("not same at ",i )
        exit()
print("same")

# x = list(set(map(tuple,L)))
# print(len(L), len(x))


#### categorical values
##    0 -- age>35  
##    1 -- age25-35
##    2 -- age<25

F_age = []

for i in range(len(F_age25)):
    if F_age35[i][3] == "0":
        F_age.append(F_age35[i])
    elif F_age35[i][3] == "1" and F_age25[i][3] == "0":
        F_age.append([F_age25[i][0], F_age25[i][1], F_age25[i][2], "1"])
    else:
        F_age.append([F_age25[i][0], F_age25[i][1], F_age25[i][2], "2"])


hdr = ["DurationMonth","CreditAmount","score","age"]
with open('GermanCredit_age.csv', mode='w') as fp:
    file_writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(hdr)
    file_writer.writerows(F_age)

#### categorical values
##    0 -- age>35,   male
##    1 -- age>35,   female
##    2 -- age25-35, male
##    3 -- age25-35, female
##    4 -- age<25,   male
##    5 -- age<25,   female


F_age_gender = []

for i in range(len(F_age25)):
    if F_sex[i][3] == "0":
        if F_age35[i][3] == "0":
            F_age_gender.append([F_age25[i][0], F_age25[i][1], F_age25[i][2], "0" ])
        elif F_age35[i][3] == "1" and F_age25[i][3] == "0":
            F_age_gender.append([F_age25[i][0], F_age25[i][1], F_age25[i][2], "2"])
        else:
            F_age_gender.append([F_age25[i][0], F_age25[i][1], F_age25[i][2], "4"])
    else:
        if F_age35[i][3] == "0":
            F_age_gender.append([F_age25[i][0], F_age25[i][1], F_age25[i][2], "1" ])
        elif F_age35[i][3] == "1" and F_age25[i][3] == "0":
            F_age_gender.append([F_age25[i][0], F_age25[i][1], F_age25[i][2], "3"])
        else:
            F_age_gender.append([F_age25[i][0], F_age25[i][1], F_age25[i][2], "5"])


hdr = ["DurationMonth","CreditAmount","score","age_gender"]
with open('GermanCredit_age_gender.csv', mode='w') as fp:
    file_writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(hdr)
    file_writer.writerows(F_age_gender)