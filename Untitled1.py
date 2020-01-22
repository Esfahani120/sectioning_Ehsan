#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py  #watch:   https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def sectioning():

    #sectioning

    
    df_sec_X=df.groupby('X').apply(pd.DataFrame.to_numpy)  #location of the section is: sec_loc[...]
    df_sec_Y=df.groupby('Y').apply(pd.DataFrame.to_numpy)
    return df_sec_X, df_sec_Y

def sec_plot(dfm):
    #### 2D data visualization
    ### #sec_plot(sectioning()[0:X-dir|1:Y-dir][location]
    dfm0=pd.DataFrame(dfm)
    phi=np.reshape(dfm0[3].to_numpy(),(129,257))
    plt.imshow(phi, extent=[0, 4, 0, 2], origin='lower', cmap='jet')
    plt.show()
    # plt.colorbar()
    return


# In[4]:


hdf= h5py.File(r'E:\baskar\data_set_microst\bikhodi.h5','w')

data=h5py.File(r'E:\baskar\data_set_microst\data_chi_3.400_phi_0.500.h5','r')
keys_list=list(data.keys())
sec_loc=np.linspace(0.0,4.0,257)
for timestep in range(2):  #use:range(len(keys_list)) for full data
    dset=data[keys_list[timestep]] #keys_list shape is (83,)
    # list(dset.items())
    # connectivity=np.array(dset.get("elem_connectivity"))  # 8388608 (256*256*128) element with 8 nodes each
    ncoord=np.array(dset.get('node_coords')) # 8520321 (257*257*129) nodes with 3 (xyz) (4*4*2)
    phi_values=np.array(dset.get('node_data').get('phi')) # 8520321 values for phi

    ###change numpy arrays to pandas to make calculation easier
    df_xyz=pd.DataFrame(ncoord,columns=['X','Y','Z'])
    df_phi=pd.DataFrame(phi_values,columns=['phi'])
    df_NN=pd.DataFrame(np.arange(len(df_phi)),columns=['NN'])
    df=pd.concat([df_xyz,df_phi,df_NN],axis=1,sort=False)


    # sec_plot(sectioning()[0][sec_loc[50]])  #sec_plot(sectioning()[0:X-dir|1:Y-dir][location]

#     with h5py.File(r'E:\baskar\data_set_microst\bikhodi.h5','w') as hdf:
    G1=hdf.create_group('G_/'+keys_list[timestep])
    tmp0=sectioning()[0]
    tmp1=tmp0.to_numpy()
    tmp2=sectioning()[1]
    tmp3=tmp2.to_numpy()
    for j in range(len(sec_loc)):
        G1.create_dataset('s_X_'+str(sec_loc[j]),data=tmp1[j])
        G1.create_dataset('s_Y_'+str(sec_loc[j]),data=tmp3[j])
data.close()
hdf.close()


# In[5]:


sec_plot(sectioning()[0][sec_loc[50]])  #sec_plot(sectioning()[0:X-dir|1:Y-dir][location]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




