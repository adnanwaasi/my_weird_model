import fastf1
import matplotlib.pyplot as plt 
import fastf1.plotting
fastf1.plotting.setup_mpl(misc_mpl_mods=False, color_scheme='fastf1')
session=fastf1.get_session(2021,"Monza","Q")
session.load()
fast_leclerc=session.laps.pick_drivers("HAM").pick_fastest()


leclerc_car_data=fast_leclerc.get_car_data()
t=leclerc_car_data["Time"]
vCar=leclerc_car_data["Speed"]

fig,ax=plt.subplots()
ax.plot(t,vCar,label="fastest")
ax.set_xlabel("time")
ax.set_ylabel("speed km/h")
ax.set_title("HAMILTON is ")
ax.legend()
plt.show()
