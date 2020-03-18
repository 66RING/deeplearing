import numpy as np

def loss(w,b,points):
    sum = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        sum += (w*x+b-y)**2
    return sum/float(len(points))

# w = w- lr*dl/dw
# b  = b- lr*dl/db
def gwb(w,b,lr,points):
    N = float(len(points))
    gw = 0
    gb = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        gw += (2/N)*((w*x+b)-y)*x
        gb += (2/N)*((w*x+b)-y)
    nw = w - lr*gw
    nb = b - lr*gb
    return nw,nb

def newwb(w,b,points,lr,num_iterations):
    for i in range(0,num_iterations):
        w,b = gwb(w,b,lr,points)
    return w,b

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    lr = 0.0001
    w = 0
    b = 0
    num_iterations = 1000
    w,b = newwb(w,b,points,lr,num_iterations)
    print("after:",num_iterations,"w:",w,"b:",b,"error:",loss(w,b,points))

if __name__ == "__main__":
    run()
