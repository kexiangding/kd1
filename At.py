import time

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
import Regression
#import Resnet8
#a=0
def funA(a):
    a()
    print ('funA')
def funB(b):
#    b()
    print ('funB')
def funD(d):
    print ('funD')
@funA
#@funB
def funC():
    print ('funC')
#funC()
#result:
def log(func):
    def wrapper():
        print('log开始 ...')
        func()
        print('log结束 ...')
    return wrapper
@log
def test():
    print('test ..')
#test()
def log(func):
    def wrapper():
        print('log开始 ...')
        func()
        print('log结束 ...')
    return wrapper
@log
def test1():
    print('test1 ..')
def test2():
    print('test2 ..')
#print(test1.__name__)
#print(test2.__name__)
test1()
test2()
from functools import wraps
def log(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        print('log开始 ...',func.__name__)
        ret = func(*args,**kwargs)
        print('log结束 ...')
        return ret
    return wrapper
@log
def test1(s):
    print('test1 ..', s)
    return s
@log
def test2(s1, s2):
    print('test2 ..', s1, s2)
    return s1 + s2
test1('a')
test2('a','bc')

def funA(fn):
    print('A')
    # 输出A
    fn() # 执行传入的fn参数　　输出B
    return time.time_ns()
    # 返回给funB
 #下面装饰效果相当于：funA(funB)，
 #7 funB将会替换（装饰）成该语句的返回值；
 #8 由于funA()函数返回fkit，因此funB就是fkit
@funA
def funB():
    print('B')
print(funB) # 此时再运行funB函数输出的是fkit

start=funB

def time1(func):
#    start=time.time_ns()
    print(time.time_ns())#.ctime())
    return func()
@time1
# 从这里可以看出@time 等价于 time(xxx()),但是这种写法你得考虑python代码的执行顺序
def xxx():
    for i in range(100000):
        a=1
#    end = time.time_ns()
    print(time.time_ns())  # .ctime())
end=time.time_ns()
interval = end - start
print(interval)
