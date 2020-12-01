# --------------


#Importing header files
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import ztest
from scipy.stats import chi2_contingency
import warnings


# In[2]:

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  

# Critical Value
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1


# In[3]:


#Reading file
data = pd.read_csv(path)
data.head(5)
#Code starts here

# In[6]:


data_sample = data.sample(n = sample_size, random_state = 0)


# In[7]:


#Calcualte Mean and std for the sample for installment
sample_mean = data_sample['installment'].mean()
sample_std  = data_sample['installment'].std()
print (sample_mean, sample_std)


margin_of_error = z_critical * sample_std / math.sqrt(sample_size) #or multiply by *0.5



#Confidence Intervals
confidence_interval = (round(sample_mean - margin_of_error,2), 
                       round(sample_mean + margin_of_error,2))
print(confidence_interval)


true_mean = round(data['installment'].mean(), 2)
print(true_mean)  # it lies withing the confidence interval


sample_sizes_array = [20, 50, 100]



# In[30]:


for i in range(len(sample_sizes_array)):
    #intialize a blank list
    
    list_s_mean = []
    list_s_size = []
    
    for j in range(1000):
        sap_mean = data['installment'].sample(n = sample_sizes_array[i]).mean()
        sap_size = len(data['installment'].sample(n = sample_sizes_array[i]))
        #append to the list
        list_s_mean.append(sap_mean)
        list_s_size.append(sap_size)


# In[34]:


fig, ax = plt.subplots(1,1)
ax.hist(list_s_mean)
ax.set_title("(1000 Random Sampled Means")
plt.show()


# In[43]:


# one sided t test for Intrest Rates
data['int.rate'] = data['int.rate'].str.replace('%', '') #.map(lambda x: str(x)[:-1])
data.head()

#convert String to float
data['int.rate'] = data['int.rate'].astype(float)

z_statistic_1, p_value_1 = ztest(x1 = data[data['purpose'] == 'small_business']['int.rate'], value = data['int.rate'].mean(), alternative = 'larger')


# In[64]:


print('z_stas1: {}'.format(z_statistic_1))
print('p_val2: {}'.format(p_value_1))


# In[59]:


#2 side t Test for Defaulters vs non defaulters
z_statistic_2, p_value_2 = ztest(x1 = data[data['paid.back.loan'] == 'No']['installment'], 
                                 x2 = data[data['paid.back.loan'] == 'Yes']['installment'], alternative = 'two-sided')


# In[65]:


print('z_stas2: {}'.format(z_statistic_2))
print('p_val2: {}'.format(p_value_2)) #reject null, there is a statisticall significant difference between installments made by people who paid back and those that havent.


# In[88]:


#Chi squre test
counts_loan_stats = pd.crosstab(data['paid.back.loan'],'purpose')
print(counts_loan_stats)

chi2, p, dof, ex = stats.chi2_contingency(counts_loan_stats)
print("Chi-square statistic = ",chi2)
print("p-value = ", p)


# In[90]:


if chi2 > critical_value:
    print('Reject H0')
else:
    print('Accept H0')
    


