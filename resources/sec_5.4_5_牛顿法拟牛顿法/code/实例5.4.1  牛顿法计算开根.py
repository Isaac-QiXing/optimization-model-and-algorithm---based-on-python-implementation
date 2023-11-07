#!/usr/bin/env python
# coding: utf-8

# 实例5.4.1  牛顿法计算开根

# In[1]:


import decimal 
x = 2.0
ite = 20 
a0 = 1.0  
bit_pre = int(1E3)
decimal.getcontext().prec = bit_pre 
a  = decimal.Decimal(a0)
xx  = decimal.Decimal(x)
half = decimal.Decimal(0.5)
for k in range(ite):     
    print('ite: ',k)
    print(" {:.500f}".format(a))
    a = half*(a + xx/a)


# In[ ]:




