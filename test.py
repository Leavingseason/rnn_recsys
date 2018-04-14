'''
@author: v-lianji

this 'test' does not mean that 'test'
I know you are expecting the 'test' as inference scripts... However, it is not.

'''



class A(object):
    def __init__(self): 
        self.name = 'A'
        self.echo()
    def echo(self):
        print('A')
        
class B(A):
    def __init__(self):
        super().__init__()  # ok, replace echo() in A with that in B
        self.a = 45
        pass 
    def echo(self):
        print('B')
        print(self.name)


b = B()