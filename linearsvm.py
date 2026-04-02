import numpy as np
def svm_lossv(W,X,y,reg):
    #全向量化的SVM损失与梯度计算
    num_train=X.shape[0]
    scores=X.dot(W)
    correct_scores=scores[np.arange(num_train),y].reshape(-1,1)
    margin=np.maximum(0,scores-correct_scores+1.0)
    margin[np.arange(num_train),y]=0
    loss=np.sum(margin)/num_train+reg*np.sum(W*W)
    margin[margin>0]=1
    count=np.sum(margin,axis=1)
    margin[np.arange(num_train),y]-=count
    dW=(X.T).dot(margin)/num_train+2*reg*W
    #梯度计算
    return loss,dW

class linearSVM:
    def __init__ (self):
        self.W=None
    def train(self,X,y,learning_rate=1e-3,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
        num_train,dim=X.shape
        num_classes=np.max(y)+1

        if self.W is None:
            self.W=0.001*np.random.randn(dim,num_classes)
        
        loss_history=[]
        for it in range(num_iters):
            batch_idx=np.random.choice(num_train,batch_size,replace=True)
            X_batch=X[batch_idx]
            y_batch=y[batch_idx]

            loss,grad=svm_lossv(self.W,X_batch,y_batch,reg)
            loss_history.append(loss)

            self.W-=learning_rate*grad
            if verbose and it%100==0:
                print('迭代%d/%d:s损失Loss=%f'%(it,num_iters,loss))
            
        return loss_history
    def predict(self,X):
        scores=X.dot(self.W)
        y_pred=np.argmax(scores,axis=1)
        return y_pred
        
