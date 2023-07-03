import numpy as np
import pandas as pd
from mpi4py import MPI
import sys
import os

def chunks(lst, n_grp):
	size = int(len(lst)/n_grp)
	for i in range(0, len(lst), size):
		yield lst[i:i + size]

def mapper(filename):
	tmp = pd.read_csv(data_dir+filename)
	partnum = filename.replace('data', '').replace('.txt','')
	tmp = pd.read_csv(data_dir+filename, header=None)
	tmp[1] = 1
	unique_values = tmp[0].unique()
	save_files = []
	for val in unique_values:
		uni_data = tmp.loc[tmp[0]==val]
		save_file = 'd'+partnum+'_key_'+str(val)+'_.npy'
		save_files += [save_file]
		np.save(save_dir+save_file, uni_data.values)   
	return save_files


def reducer(files_k):
	freq_k = 0
	for file_k in files_k:
		data_k = np.load(save_dir+file_k)
		freq_k += data_k.shape[0]
	return freq_k

data_dir = '/gpfs/projects/AMS598/Projects2022/project1/'
save_dir = '/gpfs/home/sowu/project/p2022/tmpdata/'

def run():
	comm = MPI.COMM_WORLD
	myrank = comm.rank
	p = comm.size

	if comm.rank ==0:
	    allfiles = os.listdir(data_dir)
	    allfiles = np.array(allfiles)[pd.Series(allfiles).str.contains("txt").values]
	    allfiles = allfiles[~pd.Series(allfiles).str.startswith(".").values]
	    allfiles = np.sort(allfiles)
	    myfiles = ['_'.join(ifiles) for ifiles in list(chunks(allfiles, p)) ]
	else:
	    myfiles = None

	myfiles = comm.scatter(myfiles, root=0)
	#print('rank %d: ', myrank)
	myfiles = myfiles.split('_')

	my_mapped_files = []
	for myfile in myfiles:
	    map_output = mapper(myfile)
	    my_mapped_files += map_output;

	mapped_files_gathered = comm.allgather(my_mapped_files)
	mapped_files = [];
	for i in mapped_files_gathered: mapped_files += i;

	comm.Barrier()

	file_keys = pd.Series(mapped_files).str.split('_', expand=True)
	file_keys.index = mapped_files
	file_keys.iloc[:,2] = file_keys.iloc[:,2].astype('int')
	unique_keys = file_keys.iloc[:,2].unique()
	nkey_per_p = len(unique_keys)/p;
	my_keys = unique_keys[(unique_keys>=int(myrank*nkey_per_p)) & (unique_keys<int((myrank+1)*nkey_per_p))]

	my_dict = {}
	for k in my_keys:
	    files_k = file_keys.loc[file_keys.iloc[:,2]==k].index
	    freq_k = reducer(files_k)
	    my_dict[k] = [freq_k]
	    #print(freq_k)
	my_df = pd.DataFrame.from_dict(my_dict, orient='index')
	
	reduced_keys = comm.gather(my_df, root=0)

	if myrank ==0:
	    output = pd.concat(reduced_keys).sort_values(0, ascending=False)
	    print(output.iloc[:5,])

if __name__ == "__main__":
	run();


