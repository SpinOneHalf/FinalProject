from collections import deque
import matplotlib.pyplot as plt
from matplotlib import animation
import multiprocessing as mp
import numpy as np
def run(que:mp.Queue):
        #Create plots
        fig, ax = plt.subplots(3,1)
        xdata, ydata = deque(maxlen=350), deque(maxlen=350)
        ln, = ax[0].plot([], [], 'ro',label="validation loss")
        imp1 = ax[1].imshow(np.zeros((244,255,3)))
        imp2 = ax[2].imshow(np.zeros((244,255,3)))
        fig.suptitle("losses over time")
        def update(bla):
            frame=que.get()
            if frame["type"]=="loss":
                i=frame["count"]
                loss=frame["loss"]
                xdata.append(i)
                ydata.append(loss)
                ax[0].set_xlim(min(xdata)-.5,max(xdata)+.5)
                ax[0].set_ylim(0,max(ydata))
                ln.set_data(xdata, ydata)
                return ln,imp1,imp2
            else:
                net_out=frame["net_out"]
                image_input=frame["image_input"]
                imp1.set_data(np.transpose(net_out.numpy(), (1, 2, 0)))
                imp2.set_data(np.transpose(image_input.numpy(), (1, 2, 0)))
                print("DONE")
        anim = animation.FuncAnimation(fig, update,interval=10)
        plt.show()

class PlotManager:
    def __init__(self):
        self.que=mp.Queue()
        self.main_process=mp.Process(target=run,args=(self.que,))
        self.main_process.start()
    def update_loss(self,loss,index):
        self.que.put({"type":"loss","count":index,"loss":loss})
    def update_images(self,net_out,input_image):
        self.que.put({"type":"foo","net_out":net_out,"image_input":input_image})
    def __del__(self):
        self.main_process.kill()
if __name__=="__main__":
    import numpy as np
    import time
    test_manager=PlotManager()
    for i in range(100):
        test_manager.update_loss(np.random.rand(),i)
        print("HERE")
        time.sleep(.1)
    del test_manager