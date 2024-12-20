# NOTES:
#   Intro to Step 5 - blank file in not needed, Python creates it.
#   Is there another way to describe expected RMSE? 
# What is the lmfit.Minimizer doing relative to create_calibfile?
# Does running with calb_combos lead to lots of duplicate RMSE?
# or does running once with wide range lead to best?
# 
# clock drift not mentioned in documentation
# 
# Aside:
#   try s0_2 with calibration_answers
#   consider making python visualization/check of fits
# 
# 
# Check if Alt Calib actually puts things in data/sample_data/ 
# 
# 
# Why are test_LK.py and LK.py so different?
# 
# Add logic to crop only if needed? 
# or is there some other advantage to copying it to this folder?
# D:\U\Glacier\GD_ICEH_iceHabitat\output\cam4\oblique\20190724
# copies usually get deleted


file_path = osp.dirname(osp.realpath(__file__))
print(file_path)



# 
# 
# #to set up:
camname = camnames[0]
# for counter, wsp in enumerate(workspaces, start = 1):
#   print(counter)
#   print(wsp)
# 
counter=1
wsp=workspaces[0]
ws_source=wsp
ws_target=tgw


plt.imshow(mask1)
plt.show()
plt.imshow(mask)
plt.show()
np.unique(mask1)
np.unique(mask)





plot_ws='G:\\Glacier\\GD_ICEH_iceHabitat\\output\\cam4\\oblique\\20190724\\plots'
plot_ws='G:\\Glacier\\GD_ICEH_iceHabitat\\output\\cam4\\oblique\\20190725\\plots'
plot_ws='G:\\Glacier\\GD_ICEH_iceHabitat\\output\\cam4\\oblique\\20190726\\plots'
track_len=2
track_len_sec=60




