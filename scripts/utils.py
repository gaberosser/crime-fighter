__author__ = 'gabriel'


def shutdown_decorator(func, *args, **kwargs):

    def wrapper():
        func()
        with open('/home/gabriel/signal/shut_me_down_goddamnit', 'w') as f:
            pass

    return wrapper