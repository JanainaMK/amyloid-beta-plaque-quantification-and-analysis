import h5py
import numpy as np

file = h5py.File('evaluation/localisation-proper.hdf5', 'r')

p_matrix = np.zeros((3, 4, 4))
r_matrix = np.zeros((3, 4, 4))
dto_i = {'dto0.0': 0, 'dto0.03': 1, 'dto0.04': 2, 'dto0.05': 3}
ms_i = {'ms5.0': 0, 'ms10.0': 1, 'ms15.0': 2}
ks_i = {'ks0': 0, 'ks11': 1, 'ks21': 2, 'ks31': 3}
for param_string in file:
    params = param_string.split('-')
    res = file[param_string].attrs
    p_matrix[ms_i[params[2]], dto_i[params[0]], ks_i[params[1]]] = res['precision']
    r_matrix[ms_i[params[2]], dto_i[params[0]], ks_i[params[1]]] = res['recall']
    print(param_string.rjust(20), 'precision:{:.3f}\trecall:{:.3f}'.format(res['precision'], res['recall']))

p_matrix = np.round(p_matrix, decimals=3)
r_matrix = np.round(r_matrix, decimals=3)

print(p_matrix)
print(r_matrix)
file.close()
