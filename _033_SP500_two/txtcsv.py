import pandas as pd
import os
root_dir = os.getcwd()
print( root_dir )
file = pd.read_csv(root_dir + '/log_search.txt')
new_csv_file = file.to_csv(root_dir + '/log_search.csv')

file2 = pd.read_csv(root_dir + '/log_final.txt')
new_csv_file = file2.to_csv(root_dir + '/log_final.csv')