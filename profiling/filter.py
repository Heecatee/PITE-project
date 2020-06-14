
with open('files.txt') as files, open('prof.txt') as profs:
    for prof in profs.readlines():
        for filee in files.readlines():
            if prof.find(str(filee[:-1]))>=0:
                print(prof[:-1])
        files.seek(0,0)