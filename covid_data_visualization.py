#Data visualization
import matplotlib.pyplot as plt

#plot known data
day_new=np.arange(1,10)
plt.plot(day_new,timeseries_data)
#plot predicted data
day_pred=np.arange(10,20)
plt.plot(day_pred,lst_output)
