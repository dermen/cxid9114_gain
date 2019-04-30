


plt.figure()
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(-900,900)  # work in pixel units
ax.set_ylim( 900,-900)
ax.set_facecolor('dimgray')
# you might want 64 panels for the cspad

imshow_arg = {"vmin":0, "vmax":1, "interpolation":'none', "cmap":'gnuplot'}
## plot the asics
for i in range(64): 
    geom_help.add_asic_to_ax( ax=ax, I=asic64[i], 
                    p=p64[i], fs=f64[i],ss=s64[i], s="", patches=P, **imshow_arg)

ax = plt.gca()
# show a circle for debugging centering
circ = plt.Circle(xy=(0,0), radius=70, fc='none', ec='C2', ls='dashed')
ax.add_patch(circ)
plt.show()



for i in range(64): 

    img = asic64[i].copy()
    mask = np.any(Malls[i],axis=0)
    img[mask] = np.nan
    geom_help.add_asic_to_ax( ax=ax, I=img, 
                    p=p64[i], fs=f64[i],ss=s64[i], s='', **imshow_arg)








1084/139: pwd
1084/140: clear
1084/141: pwd
1084/142: clear
1084/143: figure(2)
1084/144: ax = gca()
1084/145: ax.set_aspect("auto")
1084/146: f = gcf)_
1084/147: fig = gcf()
1084/148: fig.get_size_inches()
1084/149: subp = {'left':.07, 'bottom':0.06, 'right':.97, 'top':.97}
1084/150: subp
1084/151: subplots_adjust(**subp)
1084/152: subp = {'left':.07, 'bottom':0.06, 'right':1, 'top':1}
1084/153: subp = {'left':.05, 'bottom':0.05, 'right':1, 'top':1}
1084/154: savefig("test.png", dpi=150)
1084/155: pwd
1084/156: fig.set_size_inches(array([12.46,  6.65]))
1084/157: ax = gca()
1084/158: ax.images
1084/159: len( ax.images)
1084/160: ax.get_xlim()
1084/161: ax.set_xlim((14.193155793403356, 873.3295699909963))
1084/162: ax.get_ylim()
1084/163: ax.set_ylim((35.81022017425094, -381.0165570529714))
1084/164: %hist
1084/165: ax.clear()
1084/166:
for i in range(64): 
    img = asic64[i].copy()
    mask = np.any(Malls[i],axis=0)
    img[mask] = np.nan
    geom_help.add_asic_to_ax( ax=ax, I=img, 
                    p=p64[i], fs=f64[i],ss=s64[i], s='', **imshow_arg)


