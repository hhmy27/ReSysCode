import time

def timer(func):
    def wrapper(*args,**kwargs):
        s=time.time()
        res=func(*args,**kwargs)
        e=time.time()
        print(f'Func : {func.__name__} over, run time : {e-s:0.4f}s')
        return res
    return wrapper