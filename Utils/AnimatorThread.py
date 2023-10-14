import threading
from IPython import display
from d2l import torch as d2l

"""只是参考"""

class Animator:
    def add(self, x, y):
        '''...内容省略...'''
        if 'display' in globals():  # 非Notebook环境不调用display()
            # display:ipython = globals()['display']
            display.display(self.fig)
            display.clear_output(wait=True)


def train(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    # animator = Animator(...)
    def train_thread():
        for epoch in range(num_epochs):
            '''...原有的训练过程代码...'''
            if 'display' not in globals():
                print('epoch {0}: train loss {1}, train accuracy {2}, test accuracy {3}.'.format(
                    epoch, train_metrics[0], train_metrics[1], test_acc
                ))   # 控制台输出
                d2l.plt.draw()   # 更新绘图
        train_loss, train_acc = train_metrics
        '''...原有的训练过程代码...'''
    if 'display' not in globals():
        th = threading.Thread(target=train_thread, name='training')
        th.start()
        d2l.plt.show(block=True)   # 显示绘图
        th.join()