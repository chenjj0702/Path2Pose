import pandas as pd
import numpy as np

cont = np.array([1, 2, 3, 4])
cont = pd.DataFrame(cont, columns=['S1'])
name = 'a.xlsx'
# cont.to_excel(name, index=False, sheet_name='1')
#
cont2 = np.array([2, 3, 4, 5])
cont2 = pd.DataFrame(cont2, columns=['S2'])
data = pd.read_excel(name)
data['S2'] = cont2
data.to_excel(name, index=False, sheet_name='1')
