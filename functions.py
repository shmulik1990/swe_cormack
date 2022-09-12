#%% Functions
import numpy as np
import matplotlib.pyplot as plt
import time

# bed level functions


def flin(x,zmax,L):
    y1=(zmax-zmax/L*x)
    return(y1)
def fsin(x,zmax,L):
    y2=(zmax/2+zmax/2*np.cos(x/L*np.pi))
    return(y2)
def fexp(x,zmax,L,f):
    y3=(np.exp(-x*2*f)*zmax*(1-x/L))
    return(y3)
def fnexp(x,zmax,L,f):
    y4=(-np.exp(x*f))*(-zmax)*(1-x/L)
    return(y4)
def fElinexp(Zmax,Epot,XHS,f):
    for i in np.linspace(Zmax/5,Zmax*5,num=10000):
        zexp=fexp(x,i,XHS,f)
        di=Epot-np.round(zexp.cumsum()[-1],2)/XHS
        #print(di)
        if abs(di)<0.01:
            zfinal=i
    return(zfinal)

def fElinnexp(Zmax,Epot,XHS,f):
    for i in np.linspace(Zmax/5,Zmax*5,num=10000):
        zexp=fnexp(x,i,XHS,f)
        di=Epot-np.round(zexp.cumsum()[-1],2)/XHS
        #print(di)
        if abs(di)<0.01:
            zfinal=i
    return(zfinal)



def SWE(oldrun,path_in,tname,Hname,Qname,name,x,hini,
        tend,ts,tss,Pts,crstart,dtmax,
        hgp,n,hmin,save,path_out,rst,plotQ):
    
    #variables
    ## oldrun == True or False (if a previous run has been saved and should be continued)
    ## path_in== STRING (absolute path to saved runs)
    ## tname,Hname,Qname== STRING (names of .npy dics)
    ## x== NUMPY ARRAY (horizontal resolution of numeric grid/ computation points)
    ## hini== FLOAT (initial water depth, everywhere equal)
    ## tend== FLOAT (End computation point in time in seconds)
    ## ts== FLOAT (timestep for saving and plotting, in seconds)
    ## dtmax== FLOAT (maximum allowable timestep)
    ## hgp== NUMPY ARRAY (vertical distribution of computational points)
    ## tss== NUMPY ARRAY (timestamps in seconds of rainfall time series)
    ## Pts== NUMPY ARRAY (rainfall intensities at tss in m/s)
    ## crstart== FLOAT (usually 0.5, see Liang et al. 2006)
    ## n== FLOAT (Manning's n)
    ## hmin== FLOAT (minimum allowable water depth)
    ## beta== FLOAT (usually 1.0, vertical distribution of velocity)
    beta=1
    ## save== LOGICAL (save results or not)
    ## path_out== STRING (Absolute path to directory to save)
    ## plotQ== Logical (if true then Q at the lower end of the hillslope will be plotted every ts seconds)
    
    
    dx=x[-1]/(len(x)-1)
    dh=abs((hgp[2:]-hgp[0:-2])/(2*dx))
    dl=dx/np.cos(np.arctan(dh/dx))
    dl=np.concatenate(([dl[0]],dl,[dl[-1]]))
    
    Qout=[]
    tout=[]
    tout.append(0)
    
    ###### first we define functions which are called during the time marching:
    def dtcalc(dl,q,H,Crpreset):
        dt=Crpreset*dl/np.nanmax(q/H+np.sqrt(9.81*H))
        return(dt)

    def Fcalc(X1,X2):
        F1P=X2
        F2P=beta*X2**2/X1+9.81*X1**2/2
        return(F1P,F2P)

    def Scalc(X1,X2,P,typ):
        S1P=np.zeros(len(X1))+P
        dhdx=abs((hgp[2:]-hgp[0:-2])/(2*dx))  #central differences
        if typ=="forward":
            dhdx=np.concatenate(([dhdx[0]],dhdx,[dhdx[-1]]))
        elif typ=="backward":
            dhdx=np.concatenate(([dhdx[0]],dhdx,[dhdx[-1]]))
        Ix=dhdx*X1
        Ie=n**2*X2**2/X1**(7/3)
        dI=Ix-Ie
        S2P=9.81*(dI)
        return(S1P,S2P)

    def TVD(x1,x2,Cr):
        X=np.array((x1,x2))
        rpos=np.zeros(len(x1)-2)
        rneg=rpos.copy()
        dXus=X[:,2:]-X[:,1:-1]   #take both rows but not all columns (ghost points)
        dXds=X[:,1:-1]-X[:,:-2]
        dXus[dXus==0]=1e-99
        dXds[dXds==0]=1e-99
        for i in range(len(dXus[0])):
            rpos[i]=np.dot(dXus[:,i],dXds[:,i])/np.dot(dXus[:,i],dXus[:,i])
            rneg[i]=np.dot(dXus[:,i],dXds[:,i])/np.dot(dXds[:,i],dXds[:,i])
            if np.isnan(rpos[i]):
                pass
        rposminus=np.append(rpos[0],rpos[0:-1])
        rnegplus=np.append(rpos[1:],rpos[-1])
        tvd=(fG(rpos,Cr)+fG(rnegplus,Cr))*dXus-(fG(rneg,Cr)+fG(rposminus,Cr))*dXds
        return(tvd)

    def fCr(H,q,dtoverdx):
        Cr=(abs(q/H)+np.sqrt(9.81*H))*dtoverdx
        return(Cr)

    def fG(r,Cr):
        phi=np.zeros(len(r))
        for i in range(len(r)):
            phi[i]=np.nanmax(np.append(0,np.nanmin(np.append(2*r[i],1))))    
        C=np.where(Cr<=0.5,Cr*(1-Cr),0.25)
        G=0.5*C*(1-phi)   #do not include ghost points
        return(G)
    
    
    
    if oldrun:
        tsave=np.load(path_in+tname+".npy",allow_pickle=True).item()    
        Hsave=np.load(path_in+Hname+".npy",allow_pickle=True).item()
        Qsave=np.load(path_in+Qname+".npy",allow_pickle=True).item()
        x=np.load(path_in+"x_"+"n"+".npy",allow_pickle=True)
        hgp=np.load(path_in+"Z_"+"n"+".npy",allow_pickle=True)
        if rst<0:
            dtno=list(tsave)[-1]
        else:
            dtno=rst
        t=tsave[dtno]
        print("Restart at {}".format(t))
        H0=Hsave[dtno]
        Q0=Qsave[dtno]
        told=t
        count=0
        countn=dtno+1
    else:    
        H0=np.ones((len(x)))*hini
        Q0=H0*0
        t=0
        Hsave={}
        Qsave={}
        tsave={}
        told=0
        count=0
        countn=0
        
    # Time marching
    H=H0.copy()
    Q=Q0.copy()
    tmon0=time.time()
    while t<tend:
        tmonact=time.time()
        dtmon=tmonact-tmon0
        P=np.interp(t,tss,Pts)
        qini=0 #no flow from upstream of first point
        
        count=count+1
        #set boundary values at ghost points
        dH=(H[-1]-H[-2])
        dQ=(Q[-1]-Q[-2])
        H=np.concatenate(([H[0]],H,[H[-1]+dH]))
        Q=np.concatenate(([qini],Q,[Q[-1]+dQ]))    
        dtorg=np.min(dtcalc(dl,abs(Q),abs(H),crstart))
        
          
        if dtorg>dtmax:
            dt=dtmax
        else:
            dt=dtorg
        
        
        dtoverdl=dt/dl
        dtoverdx=dt/dx
        
        F1,F2=Fcalc(abs(H),abs(Q))
        S1,S2=Scalc(H,Q,P,typ="forward")
        S1[0]=0
        S1[-1]=0
        #forward differences
        dF1=F1[2:]-F1[1:-1]
        dF2=F2[2:]-F2[1:-1]
    
        
        # 1) predictor step
        HP=H[1:-1]-dtoverdx*dF1+S1[1:-1]*dt
        qP=Q[1:-1]-dtoverdx*dF2+S2[1:-1]*dt
        
        #set boundary values at ghost points
        dH=(HP[-1]-HP[-2])
        dQ=(qP[-1]-qP[-2])
        HP=np.concatenate(([HP[0]],HP,[HP[-1]+dH]))
        QP=np.concatenate(([qini],qP,[qP[-1]+dQ]))
        
        HP[HP<hmin]=hmin
        QP[np.where(HP<hmin)]=0
        
        #insert XP into F,S to get FP,SP:
        F1P,F2P=Fcalc(abs(HP),abs(QP))
        S1P,S2P=Scalc(HP,QP,P,typ="backward")
        S1P[0]=0
        S1P[-1]=0
        
        dF1P=F1P[1:-1]-F1P[0:-2]
        dF2P=F2P[1:-1]-F2P[0:-2]
        
        # 2) corrector step
        HC=H[1:-1]-dtoverdx*dF1P+S1P[1:-1]*dt
        qC=Q[1:-1]-dtoverdx*dF2P+S2P[1:-1]*dt    
        
        #calculate local courantnumbers
        Cr=fCr(H,Q,dtoverdl)
        
        #set boundary values at ghost points
        dH=(HC[-1]-HC[-2])
        dQ=(qC[-1]-qC[-2])
        HC=np.concatenate(([HC[0]],HC,[HC[-1]+dH]))
        QC=np.concatenate(([qini],qC,[qC[-1]+dQ]))
    
        HC[HC<hmin]=hmin
        QC[np.where(HC<hmin)]=0

        #Final value not incl. ghost points
        TVDX1=TVD(H,Q,Cr[1:-1])[0,:]
        TVDX2=TVD(H,Q,Cr[1:-1])[1,:]

            
        HF=(HP[1:-1]+HC[1:-1])/2
        QF=(QP[1:-1]+QC[1:-1])/2
        HF[2:-1]=HF[2:-1]+TVDX1[2:-1]
        QF[2:-1]=QF[2:-1]+TVDX2[2:-1]
        
        HF[np.isnan(HF)]=hmin
        
        H=HF.copy()
        Q=QF.copy()
        t=t+dt
        
        
            
        
        print("Steps:{} time:{} dt:{} Hmax:{} Hmin:{}".format(
                                     count,np.round(t,6),np.round(dt,6),
                                     np.round(np.max(HF),6),np.round(np.min(HF),6)))
        if t>=told+ts or told==0:
            print("saved at {} seconds; seconds computated:{}".format(
                                                              round(t,4),
                                                              round(dtmon,2)))
            tsave[countn]=t
            Qsave[countn]=QF
            Hsave[countn]=HF
            countn=countn+1
            told=t

            if (plotQ):
                Qout.append(QF[-1])
                tout.append(t)
                plt.scatter(tout[1:],Qout)
                plt.draw()
                plt.pause(0.01)
                #print(dtorg)
            
            
            
    if save:
        #save result dicts to folder
        np.save(path_out+tname+"_n"+".npy",tsave)
        np.save(path_out+Qname+"_n"+".npy",Qsave)    
        np.save(path_out+Hname+"_n"+".npy",Hsave)
        np.save(path_out+"x"+name+"_n"+".npy",x)
        np.save(path_out+"Z"+name+"_n"+".npy",hgp)
        np.save(path_out+"t"+name+"_n"+".npy",tss)
        np.save(path_out+"P"+name+"_n"+".npy",Pts)
        f=open(path_out+"Run_out_"+tname+"_n.txt","w")
        if oldrun:
            f.write("restarted from "+tname+"\n")
            f.write("started at: {}".format(tsave[dtno])+"\n")
        else:
            f.write("initial Run: "+tname+"\n")
            f.write("started at: 0\n")
        f.write("dt save: "+str(ts)+"\n")
        f.write("Mannings n: {}".format(n)+"\n")
        f.write("ended at: {}".format(tend)+"\n")
        f.write("total runtime: {}".format(dtmon)+"\n")
        f.close()
        
    return(tsave,Qsave,Hsave)