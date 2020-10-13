import numpy as np
import matplotlib.pyplot as plt


for i_speed in range(5):
    for i_wide in range(5):
        for i_height in range(5):
            for i_rot in range(5):
                dirname = str(i_speed)+'_'+ str(i_wide)+'_'+ str(i_height)+'_'+ str(i_rot)
                heightmap = np.load('Data_newTake/' + dirname + '/heightmap.npy')
                rx = np.load('Data_newTake/' + dirname + '/rx.npy')
                ry = np.load('Data_newTake/' + dirname + '/ry.npy')
                fx = np.load('Data_newTake/' + dirname + '/fx.npy')
                fy = np.load('Data_newTake/' + dirname + '/fy.npy')
                # print(heightmap.shape)
                newmap = heightmap[160:260,160:260]

                xlocs = np.where(np.logical_and(rx>=160,rx<=260))
                ylocs = np.where(np.logical_and(ry>=160,ry<=260))
                rintersect = np.intersect1d(xlocs,ylocs)
                nrx = rx[rintersect] - 160
                nry = ry[rintersect] - 160

                xlocs = np.where(np.logical_and(fx>=160,fx<=260))
                ylocs = np.where(np.logical_and(fy>=160,fy<=260))
                fintersect = np.intersect1d(xlocs,ylocs)
                nfx = fx[fintersect] - 160
                nfy = fy[fintersect] - 160

                plt.imshow(newmap)
                plt.plot(nrx,nry)
                plt.plot(nfx,nfy)
                plt.gca().invert_yaxis()
                # plt.show()
                plt.savefig('Data_newTake/' + dirname+ '/cropped_summary.png')
                plt.clf()

                np.save('Data_newTake/' + dirname + '/cropped_heightmap.npy',newmap)
                np.save('Data_newTake/' + dirname + '/cropped_rx.npy',nrx)
                np.save('Data_newTake/' + dirname + '/cropped_ry.npy',nry)
                np.save('Data_newTake/' + dirname + '/cropped_fx.npy',nfx)
                np.save('Data_newTake/' + dirname + '/cropped_fy.npy',nfy)
                np.save('Data_newTake/' + dirname + '/rintersect.npy',rintersect)
                np.save('Data_newTake/' + dirname + '/fintersect.npy',fintersect)
