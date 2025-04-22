from multiprocessing import Process, Queue
from gui.run_gui import run_gui
import time 


def main(queue):

    for i in range(100):
        queue.put(({"pose" : [i,i,i]}))
        time.sleep(0.1)
        

if __name__ == "__main__":


    queue = Queue()
    sim_proc = Process(target=main, args=(queue,))
    vis_proc = Process(target=run_gui, args=(queue,))
    vis_proc.daemon = True

    sim_proc.start()
    vis_proc.start()

    try:
        sim_proc.join()
    except KeyboardInterrupt:
        queue.put("STOP")
    finally:
        queue.put("STOP")
        sim_proc.terminate()
        vis_proc.terminate()
