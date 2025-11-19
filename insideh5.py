import h5py

def print_structure(name, obj):
    print(name)

f = h5py.File('E06SCTL2B2023002_00544_00545_SN_12km_2023-002T22-37-52_v1.0.0.h5', 'r')
f.visititems(print_structure) # This lists everything inside the file