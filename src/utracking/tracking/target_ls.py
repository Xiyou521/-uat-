from collections import deque
import numpy as np

class TargetLS(object):
    
    def __init__(self, num_points=30):
        self.lsxs = None
        self.Plsu = None 
        self.eastingpoints_LS = deque(maxlen=num_points)
        self.northingpoints_LS = deque(maxlen=num_points)
        self.allz = deque(maxlen=num_points)
        self.N = np.identity(3)[:-1]
        
    def add_range(self,z,myobserver,update=False):
        self.allz.append(z)
        self.eastingpoints_LS.append(myobserver[0])
        self.northingpoints_LS.append(myobserver[1])
        if update:
            self.update()

    def get_pred(self):
        if self.lsxs is None:
            raise ("You need to update the model at least once before getting the predictions")
        else:
            return self.lsxs[[0,2]]

        
    def update(self,myobserver,dt=0.04):
        numpoints = len(self.allz)
        if numpoints > 3:
            # N can be precomputed
            N = self.N
            # P
            P = np.array([self.eastingpoints_LS,self.northingpoints_LS])
            # A
            A = np.full((numpoints,3),-1, dtype=float)
            A[:,0] = P[0]*2
            A[:,1] = P[1]*2
            #b 
            allz_np = np.array(self.allz)
            b = (np.diag(np.matmul(P.T,P))-(allz_np*allz_np)).reshape(numpoints, 1)

            try:
                # N*(A.T*A).I*A.T*b
                self.Plsu = np.matmul(np.matmul(np.matmul(N,np.linalg.inv(np.matmul(A.T,A))), A.T), b)
            except:
                print('WARNING: LS singular matrix')
                try:
                    self.Plsu = np.matmul(np.matmul(np.matmul(N,np.linalg.inv(np.matmul(A.T,A+1e-6))), A.T), b)
                except:
                    pass
            # Finally we calculate the depth as follows
#                r=np.matrix(np.power(allz,2)).T
#                a=np.matrix(np.power(Plsu[0]-eastingpoints_LS,2)).T
#                b=np.matrix(np.power(Plsu[1]-northingpoints_LS,2)).T
#                depth = np.sqrt(np.abs(r-a-b))
#                depth = np.mean(depth)
#                Plsu = np.concatenate((Plsu.T,np.matrix(depth)),axis=1).T
            #add offset
#                Plsu[0] = Plsu[0] + t_position.item(0)
#                Plsu[1] = Plsu[1] + t_position.item(1)
#                eastingpoints = eastingpoints + t_position.item(0)
#                northingpoints = northingpoints + t_position.item(1)
            #Error in 'm'
#                error = np.concatenate((t_position.T,np.matrix(simdepth)),axis=1).T - Plsu
#                allerror = np.append(allerror,error,axis=1)
        if self.lsxs is None:
            self.lsxs = np.array([myobserver[0],0,myobserver[1],0,0])

        if self.Plsu is not None:
            ls_orientation = np.arctan2(self.Plsu[1]-self.lsxs[2],self.Plsu[1]-self.lsxs[0])
            ls_velocity = np.array([(self.Plsu[0]-self.lsxs[0])/dt,(self.Plsu[1]-self.lsxs[1])/dt])
            self.lsxs  = np.array([self.Plsu.item(0),ls_velocity.item(0),self.Plsu.item(1),ls_velocity.item(1),ls_orientation.item(0)])