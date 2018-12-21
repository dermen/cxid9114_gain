import os

root = "/reg/d/psdm/cxi/cxid9114/scratch/dermen/hit_finding"
spots_f = os.path.join( root, "spots2.phil")
loc_f = os.path.join( root, "test.loc")
for run in range(33,77):
    out_d = os.path.join( root, "run%d"%run )
    if not os.path.exists( out_d):
        os.makedirs( out_d)

    loc = open(loc_f,"r").read()
    loc = loc.replace("run = 62","run = %d"%run)
    
    new_loc_f = "%s/%s"%( out_d, "loc_run%d.txt"%run)
    new_loc = open(new_loc_f, "w")
    new_loc.write(loc)
    new_loc.close()

    out_f = "%s/%s"%(out_d, "strong_run%d.pkl"%run)
    cmd = ["dials.find_spots",new_loc_f, spots_f, "output.reflections=%s"%out_f ]
    
    bsub = ["bsub", 
        "-J %d.spt"%run,
        "-q psanaq",
        "-o /reg/d/psdm/cxi/cxid9114/scratch/dermen/hit_finding/logs/%d.out"%run]
    bsub = " ".join(bsub)
    cmd = " ".join( cmd)
    os.chdir(out_d)
    print(bsub + " " + cmd)
    os.system( bsub + "  " + cmd)
    os.chdir(root)
