import pandas as pd
import os

root_dir = os.getcwd()
print( dirname )

def get_folders(loc):
    current_directory = dirname+ '/' + loc
    list=[]
    for dir in os.listdir(current_directory):
        if os.path.isdir(os.path.join(current_directory, dir)):
            list.append(dir)

            print(dir)


    list.sort()
    print(list)

    search_dir =   dirname+ '/' + loc
    os.chdir(search_dir)
    files = filter(os.path.isfile, os.listdir(search_dir))
    files = [os.path.join(search_dir, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))


    dict = { loc+'_folders': list}

    df = pd.DataFrame(dict)

    df.to_csv(dirname+'/' + loc +'_folders.csv',index=False)

get_folders(loc='outputs')
get_folders(loc='outputs_ind')